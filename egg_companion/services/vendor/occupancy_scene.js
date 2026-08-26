import * as THREE from 'three';
import { OrbitControls } from '/assets/OrbitControls.js?v=20260811a';

// Renders the fused voxel occupancy grid published by /api/occupancy --
// one shared "egg frame" (Y up, +Z the video0-camera-array boresight, see
// core/occupancy.py's module docstring) built by rotating each camera's
// monocular depth by its known array yaw before back-projecting.
//
// Confirmed on real hardware (Chromium GPU-process log): "Could not
// create a WebGL context ... GL_VENDOR = Disabled, Sandboxed = yes,
// BindToCurrentSequence failed" when a second, fully independent
// THREE.WebGLRenderer is constructed while knowledge_graph.js's own
// renderer is already alive -- this browser/GPU only sustains ONE live
// WebGL context, not "fewer than desktop Chrome's ~16." So this MUST
// borrow that exact renderer via window.__eggGraph (moving its <canvas>
// into this page's container while /vision is active, and handing it
// back on navigating away) rather than ever constructing its own.

// Shared by both render paths (WebGL and the 2D canvas fallback below) so
// the toolbar's Voxel +/- buttons work regardless of which one is active.
let voxelScaleMultiplier = 1.0;
const VOXEL_SCALE_MIN = 0.2, VOXEL_SCALE_MAX = 5.0, VOXEL_SCALE_STEP = 1.25;
window.addEventListener('egg:occupancy-voxel-scale', event => {
  const direction = event.detail?.direction || 1;
  voxelScaleMultiplier = Math.max(
    VOXEL_SCALE_MIN,
    Math.min(VOXEL_SCALE_MAX, direction > 0 ? voxelScaleMultiplier * VOXEL_SCALE_STEP : voxelScaleMultiplier / VOXEL_SCALE_STEP),
  );
});

// Increasing the Voxel +/- control must consolidate neighboring points
// into larger, gap-filling blocks -- not just inflate each native voxel's
// own footprint in place, which only produces overlap and, when shrunk,
// visible negative space between the still-native-spaced points. Cells
// coarser than the native voxel_size_meters are re-bucketed here (color
// averaged across every native voxel that lands in the same coarser
// cell), so a coarser "voxel size" genuinely represents a lower-resolution
// resampling of the same reconstruction, the way it would if the grid
// itself were built at that resolution. Below native size there's no
// finer data to recover, so the raw (native-resolution) list is used
// as-is -- shrinking legitimately reveals the true sparsity there.
function rebucketVoxels(rawVoxels, cellSize, nativeSize) {
  if (!rawVoxels.length || cellSize <= nativeSize * 1.02) return rawVoxels;
  const buckets = new Map();
  for (const v of rawVoxels) {
    const bx = Math.floor(v.x / cellSize), by = Math.floor(v.y / cellSize), bz = Math.floor(v.z / cellSize);
    const key = `${bx}|${by}|${bz}`;
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = { x: (bx + 0.5) * cellSize, y: (by + 0.5) * cellSize, z: (bz + 0.5) * cellSize, r: 0, g: 0, b: 0, n: 0 };
      buckets.set(key, bucket);
    }
    const [r, g, b] = v.color || [0x66, 0x7e, 0xa8];
    bucket.r += r; bucket.g += g; bucket.b += b; bucket.n += 1;
  }
  const merged = [];
  for (const bucket of buckets.values()) {
    merged.push({
      x: bucket.x, y: bucket.y, z: bucket.z,
      color: [Math.round(bucket.r / bucket.n), Math.round(bucket.g / bucket.n), Math.round(bucket.b / bucket.n)],
    });
  }
  return merged;
}

// Pure-2D-canvas compatibility renderer, used whenever no real WebGL
// context is available at all (confirmed on real hardware: this
// Chromium instance runs with its GPU process launched with
// --use-gl=disabled, so THREE.WebGLRenderer fails identically for
// knowledge_graph.js's own construction too -- "GL_VENDOR = Disabled,
// Sandboxed = yes, BindToCurrentSequence failed"). Mirrors
// knowledge_graph.js's own initCanvasFallback technique exactly (same
// yaw/pitch rotation + simple perspective-divide projection), which is
// proven to render correctly in this exact browser without any GPU
// acceleration at all.
function initCanvasFallback(container) {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', 'Compatibility rendering of the fused voxel occupancy reconstruction');
  container.replaceChildren(canvas);
  const context = canvas.getContext('2d', { alpha: false });

  let rawVoxels = [], voxels = [], voxelSizeMeters = 0.1;
  let pixelRatio = 1, width = 1, height = 1;
  let zoom = 60, panX = 0, panY = 0, yaw = 0.5, pitch = -0.25;
  let target = { x: 0, y: 0, z: 0 };
  let drag = null, framed = false;
  // Egg-POV lock: instead of orbiting a pivot, the camera sits at the
  // shared fused-frame origin (0,0,0) -- per core/occupancy.py's module
  // docstring, that origin IS the rig's single modeled optical center,
  // i.e. physically the center of the 256mm ring the four cameras are
  // mounted around -- and dragging looks around from there instead of
  // orbiting. povYaw/povPitch are independent of the orbit mode's
  // yaw/pitch/target/pan so switching back restores the prior orbit view.
  let povLocked = false, povYaw = 0, povPitch = 0, povDrag = null;

  function point(p) {
    const x = p.x - target.x, y = p.y - target.y, z = p.z - target.z;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const rx = cy * x + sy * z, rz = -sy * x + cy * z, ry = cp * y - sp * rz, depth = sp * y + cp * rz;
    const perspective = Math.max(0.25, Math.min(2.4, 1 - depth / 12));
    // Screen X is mirrored (-rx) so the array reads left-to-right on
    // screen (matching how a viewer standing behind the rig reads
    // video0..video3) -- this never touches core/occupancy.py's actual
    // fusion geometry, which stays physically accurate independent of
    // display.
    return { x: width / 2 + panX - rx * zoom * perspective, y: height / 2 + panY - ry * zoom * perspective, depth, scale: perspective };
  }

  // True perspective-divide projection from a camera fixed at the origin
  // (unlike point()'s weak-perspective orbit approximation, which assumes
  // the camera is always far from target) -- appropriate here because in
  // POV mode the "camera" is inside the reconstruction, often within a
  // voxel or two of nearby surfaces, where a real 1/depth projection is
  // needed for correct scale. zoom is reused as a shared FOV knob so
  // scroll-to-zoom keeps doing something meaningful in both modes.
  function povPoint(p) {
    const cy = Math.cos(povYaw), sy = Math.sin(povYaw), cp = Math.cos(povPitch), sp = Math.sin(povPitch);
    const rx = cy * p.x + sy * p.z, rz = -sy * p.x + cy * p.z;
    const ry = cp * p.y - sp * rz, depth = sp * p.y + cp * rz;
    if (depth <= 0.05) return null; // behind the camera
    const f = height * 0.9 * (zoom / 60);
    return { x: width / 2 - (rx * f) / depth, y: height / 2 - (ry * f) / depth, depth, scale: f / depth };
  }

  function resize() {
    width = Math.max(1, container.clientWidth);
    height = Math.max(1, container.clientHeight);
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  new ResizeObserver(resize).observe(container);
  resize();

  canvas.addEventListener('pointerdown', event => {
    canvas.setPointerCapture(event.pointerId);
    if (povLocked) { povDrag = { x: event.clientX, y: event.clientY, yaw: povYaw, pitch: povPitch }; return; }
    drag = { x: event.clientX, y: event.clientY, panX, panY, yaw, pitch, pan: event.shiftKey || event.button !== 0 };
  });
  canvas.addEventListener('pointermove', event => {
    canvas.style.cursor = 'grab';
    if (povLocked) {
      if (!povDrag) return;
      const dx = event.clientX - povDrag.x, dy = event.clientY - povDrag.y;
      povYaw = povDrag.yaw - dx * 0.006;
      // Pitch is sign-flipped relative to orbit mode's vertical drag
      // (+dy, not -dy) -- looking around from inside the reconstruction
      // reads naturally as "drag down to look down", the opposite feel
      // from orbiting a target from outside it.
      povPitch = Math.max(-1.5, Math.min(1.5, povDrag.pitch + dy * 0.006));
      return;
    }
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (drag.pan) { panX = drag.panX + dx; panY = drag.panY + dy; }
    else { yaw = drag.yaw + dx * 0.007; pitch = Math.max(-1.35, Math.min(1.35, drag.pitch - dy * 0.007)); }
  });
  canvas.addEventListener('pointerup', () => { drag = null; povDrag = null; });
  canvas.addEventListener('pointercancel', () => { drag = null; povDrag = null; });
  canvas.addEventListener('contextmenu', event => event.preventDefault());
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoom = Math.max(8, Math.min(400, zoom * Math.exp(-event.deltaY * 0.001)));
  }, { passive: false });

  function rebucket() {
    voxels = rebucketVoxels(rawVoxels, voxelSizeMeters * voxelScaleMultiplier, voxelSizeMeters);
  }

  function applyPayload(payload) {
    if (!payload || !payload.enabled) { rawVoxels = []; voxels = []; return; }
    rawVoxels = payload.voxels || [];
    voxelSizeMeters = payload.voxel_size_meters || 0.1;
    rebucket();
    if (!framed && rawVoxels.length) {
      const n = rawVoxels.length;
      target = {
        x: rawVoxels.reduce((sum, v) => sum + v.x, 0) / n,
        y: rawVoxels.reduce((sum, v) => sum + v.y, 0) / n,
        z: rawVoxels.reduce((sum, v) => sum + v.z, 0) / n,
      };
      framed = true;
    }
  }

  // Real cube geometry, not a flat billboard sprite: each face's 4 local
  // corner offsets (unit cube, -0.5..0.5 per axis) plus a flat per-axis
  // shade so the cube reads as 3D without real lighting. Which 3 of 6
  // faces are camera-facing is the same for every voxel simultaneously
  // (axis-aligned cubes, one shared camera orientation), so it's computed
  // once per frame in render(), not per voxel.
  const CUBE_FACES = [
    { normal: [1, 0, 0], corners: [[.5, -.5, -.5], [.5, .5, -.5], [.5, .5, .5], [.5, -.5, .5]], shade: 0.8 },
    { normal: [-1, 0, 0], corners: [[-.5, -.5, .5], [-.5, .5, .5], [-.5, .5, -.5], [-.5, -.5, -.5]], shade: 0.55 },
    { normal: [0, 1, 0], corners: [[-.5, .5, -.5], [-.5, .5, .5], [.5, .5, .5], [.5, .5, -.5]], shade: 1.0 },
    { normal: [0, -1, 0], corners: [[-.5, -.5, .5], [-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5]], shade: 0.35 },
    { normal: [0, 0, 1], corners: [[-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], shade: 0.7 },
    { normal: [0, 0, -1], corners: [[.5, -.5, -.5], [-.5, -.5, -.5], [-.5, .5, -.5], [.5, .5, -.5]], shade: 0.9 },
  ];

  function visibleCubeFaces(cy, sy, cp, sp) {
    return CUBE_FACES.filter(face => {
      const [nx, ny, nz] = face.normal;
      const rz = -sy * nx + cy * nz;
      return sp * ny + cp * rz < 0; // faces toward the camera (see point()'s depth convention)
    });
  }

  function render() {
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = '#070d19';
    context.fillRect(0, 0, width, height);

    const project = povLocked ? povPoint : point;
    const cy = Math.cos(povLocked ? povYaw : yaw), sy = Math.sin(povLocked ? povYaw : yaw);
    const cp = Math.cos(povLocked ? povPitch : pitch), sp = Math.sin(povLocked ? povPitch : pitch);
    const visibleFaces = visibleCubeFaces(cy, sy, cp, sp);
    const voxelWorldSize = voxelSizeMeters * voxelScaleMultiplier;

    const drawables = [];
    for (const v of voxels) {
      const projected = project(v);
      if (projected) drawables.push({ v, depth: projected.depth });
    }
    drawables.sort((a, b) => b.depth - a.depth); // paint far first

    for (const d of drawables) {
      const [r, g, b] = d.v.color || [0x66, 0x7e, 0xa8];
      for (const face of visibleFaces) {
        // Each corner is a REAL world-space offset of the voxel, so it
        // goes through the same perspective/zoom projection as every
        // other point -- the cube's on-screen size and shape follow
        // actual spatial geometry (distance, zoom, viewing angle), not
        // an arbitrary flat pixel constant.
        const projected = face.corners.map(([lx, ly, lz]) => project({
          x: d.v.x + lx * voxelWorldSize, y: d.v.y + ly * voxelWorldSize, z: d.v.z + lz * voxelWorldSize,
        })).filter(Boolean);
        if (projected.length < face.corners.length) continue; // a corner is behind the POV camera
        context.fillStyle = `rgb(${Math.round(r * face.shade)},${Math.round(g * face.shade)},${Math.round(b * face.shade)})`;
        context.beginPath();
        context.moveTo(projected[0].x, projected[0].y);
        for (let i = 1; i < projected.length; i++) context.lineTo(projected[i].x, projected[i].y);
        context.closePath();
        context.fill();
      }
    }
    requestAnimationFrame(render);
  }

  window.addEventListener('egg:occupancy-data', event => applyPayload(event.detail));
  window.addEventListener('egg:occupancy-reset', () => {
    target = { x: 0, y: 0, z: 0 }; yaw = 0.5; pitch = -0.25; zoom = 60; panX = 0; panY = 0; framed = false;
    povYaw = 0; povPitch = 0;
  });
  window.addEventListener('egg:occupancy-voxel-scale', () => rebucket());
  window.addEventListener('egg:occupancy-pov-toggle', () => {
    povLocked = !povLocked;
    if (povLocked) { povYaw = 0; povPitch = 0; }
    drag = null; povDrag = null;
    window.dispatchEvent(new CustomEvent('egg:occupancy-pov-changed', { detail: { locked: povLocked } }));
  });
  window.dispatchEvent(new CustomEvent('egg:occupancy-renderer', { detail: { mode: 'canvas-2d' } }));
  render();
}

const container = document.getElementById('occupancy-scene');

if (container) {
  let scene, camera, renderer, controls, resizeObserver, animationFrame;
  let voxelMesh = null, rangeWireframe = null;
  let lastPayload = null;
  // Egg-POV lock: the camera sits at the fused frame's origin (0,0,0) --
  // per core/occupancy.py's module docstring, that origin IS the rig's
  // single modeled optical center, i.e. physically the center of the
  // 256mm ring the four cameras are mounted around -- and dragging looks
  // around from there (FPS-style) instead of orbiting a target.
  let povLocked = false, povYaw = 0, povPitch = 0, povDrag = null;
  const DEFAULT_CAMERA_POSITION = { x: 3.2, y: 2.4, z: 4.2 };
  const DEFAULT_CONTROLS_TARGET = { x: 0, y: 0.3, z: 0 };

  const voxelGeometry = new THREE.BoxGeometry(1, 1, 1);
  const voxelMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05 });
  const tmpColor = new THREE.Color();
  const tmpMatrix = new THREE.Matrix4();

  // Nominal radius used only to pad fitCameraToScene's bounding box when
  // there are few/no voxels yet -- not physical rig geometry (see the
  // module docstring in core/occupancy.py: the fused frame's origin
  // already models the rig's single shared optical center).
  function nominalSceneRadius(maxRange) {
    return THREE.MathUtils.clamp((maxRange || 6.0) * 0.32, 0.6, 3.0);
  }

  function renderVoxels(voxels, voxelSize) {
    if (voxelMesh) { scene.remove(voxelMesh); voxelMesh.geometry.dispose(); voxelMesh = null; }
    if (!voxels.length) return;
    // voxelSize is already the fully-resolved cell size (native size *
    // voxelScaleMultiplier, applied by the caller before rebucketing) --
    // do not multiply by voxelScaleMultiplier again here.
    const size = Math.max(voxelSize, 0.01);
    voxelMesh = new THREE.InstancedMesh(voxelGeometry, voxelMaterial, voxels.length);
    voxelMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(voxels.length * 3), 3);
    voxels.forEach((voxel, index) => {
      tmpMatrix.makeScale(size * 0.92, size * 0.92, size * 0.92);
      // Mirrored X (see the 2D fallback's point() for why) so the array
      // reads left-to-right on screen in both render paths consistently.
      tmpMatrix.setPosition(-voxel.x, voxel.y, voxel.z);
      voxelMesh.setMatrixAt(index, tmpMatrix);
      const [r, g, b] = voxel.color || [0x66, 0x7e, 0xa8];
      tmpColor.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      voxelMesh.setColorAt(index, tmpColor);
    });
    voxelMesh.instanceMatrix.needsUpdate = true;
    if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
    scene.add(voxelMesh);
  }

  function renderRangeWireframe(maxRange) {
    if (rangeWireframe) {
      scene.remove(rangeWireframe);
      rangeWireframe.geometry.dispose();
      rangeWireframe.material.dispose();
      rangeWireframe = null;
    }
    if (!maxRange) return;
    const geometry = new THREE.SphereGeometry(maxRange, 24, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0x344054, wireframe: true, transparent: true, opacity: 0.14 });
    rangeWireframe = new THREE.Mesh(geometry, material);
    scene.add(rangeWireframe);
  }

  // Frame the camera on the actual scene content once real data arrives,
  // rather than trusting a single hardcoded guess to suit every room.
  let framedOnce = false;
  function fitCameraToScene(payload) {
    const voxels = payload.voxels || [];
    const radius = nominalSceneRadius(payload.max_range_meters);
    const box = new THREE.Box3();
    box.expandByPoint(new THREE.Vector3(-radius, 0, -radius));
    box.expandByPoint(new THREE.Vector3(radius, 0.5, radius));
    for (const v of voxels) box.expandByPoint(new THREE.Vector3(-v.x, v.y, v.z)); // mirrored, matches renderVoxels()
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z, 1.0);
    controls.target.copy(center);
    camera.position.set(center.x + span * 0.75, center.y + span * 0.6, center.z + span * 0.95);
    controls.update();
    framedOnce = true;
  }

  function applyOccupancy(payload) {
    lastPayload = payload;
    if (!scene) return;
    if (!payload || !payload.enabled) {
      if (voxelMesh) { scene.remove(voxelMesh); voxelMesh.geometry.dispose(); voxelMesh = null; }
      return;
    }
    const nativeSize = payload.voxel_size_meters || 0.1;
    const cellSize = nativeSize * voxelScaleMultiplier;
    renderVoxels(rebucketVoxels(payload.voxels || [], cellSize, nativeSize), cellSize);
    renderRangeWireframe(payload.max_range_meters);
    if (!framedOnce && !povLocked) fitCameraToScene(payload);
  }

  function resize() {
    if (!renderer) return;
    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }

  function animate() {
    animationFrame = requestAnimationFrame(animate);
    if (!povLocked) controls.update(); // manual pose while locked; see applyPovLook()
    renderer.render(scene, camera);
  }

  // Point the camera per povYaw/povPitch from a fixed position at the
  // fused frame's origin -- FPS-style look, independent of OrbitControls
  // (disabled while locked; see setPovLocked()).
  function applyPovLook() {
    if (!camera) return;
    camera.position.set(0, 0, 0);
    camera.rotation.order = 'YXZ';
    camera.rotation.set(povPitch, povYaw, 0);
  }

  function setPovLocked(locked) {
    povLocked = locked;
    if (!camera || !controls) return;
    controls.enabled = !locked;
    if (locked) {
      povYaw = 0; povPitch = 0;
      applyPovLook();
    } else {
      camera.rotation.order = 'XYZ';
      camera.position.set(DEFAULT_CAMERA_POSITION.x, DEFAULT_CAMERA_POSITION.y, DEFAULT_CAMERA_POSITION.z);
      controls.target.set(DEFAULT_CONTROLS_TARGET.x, DEFAULT_CONTROLS_TARGET.y, DEFAULT_CONTROLS_TARGET.z);
      controls.update();
    }
    window.dispatchEvent(new CustomEvent('egg:occupancy-pov-changed', { detail: { locked } }));
  }

  function onPovPointerDown(event) {
    if (!povLocked) return;
    povDrag = { x: event.clientX, y: event.clientY, yaw: povYaw, pitch: povPitch };
  }
  function onPovPointerMove(event) {
    if (!povLocked || !povDrag) return;
    const dx = event.clientX - povDrag.x, dy = event.clientY - povDrag.y;
    povYaw = povDrag.yaw - dx * 0.005;
    // Sign-flipped vs orbit mode's vertical drag -- see the 2D fallback's
    // onPovPointerMove-equivalent handler for why.
    povPitch = Math.max(-1.5, Math.min(1.5, povDrag.pitch + dy * 0.005));
    applyPovLook();
  }
  function onPovPointerUp() { povDrag = null; }

  let fallbackActive = false;

  function activate() {
    if (renderer || fallbackActive) return; // already active in some mode
    framedOnce = false;

    const shared = window.__eggGraph;
    if (!shared || !shared.renderer) {
      // No live WebGL context exists anywhere on this page (confirmed on
      // real hardware: this browser's Chromium GPU process runs with
      // --use-gl=disabled, so knowledge_graph.js's own renderer
      // construction fails identically) -- fall back to a pure 2D canvas
      // renderer instead of leaving this permanently inert, mirroring
      // knowledge_graph.js's own proven initCanvasFallback technique.
      fallbackActive = true;
      initCanvasFallback(container);
      return;
    }
    renderer = shared.renderer;
    shared.pause();
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    camera.position.set(DEFAULT_CAMERA_POSITION.x, DEFAULT_CAMERA_POSITION.y, DEFAULT_CAMERA_POSITION.z);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 0.4;
    controls.maxDistance = 60;
    controls.target.set(DEFAULT_CONTROLS_TARGET.x, DEFAULT_CONTROLS_TARGET.y, DEFAULT_CONTROLS_TARGET.z);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 0.6);
    key.position.set(3, 6, 4);
    scene.add(key);
    scene.add(new THREE.AxesHelper(0.35));

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();
    animate();

    renderer.domElement.addEventListener('pointerdown', onPovPointerDown);
    window.addEventListener('pointermove', onPovPointerMove);
    window.addEventListener('pointerup', onPovPointerUp);

    if (povLocked) applyPovLook();
    if (lastPayload) applyOccupancy(lastPayload);
  }

  function deactivate() {
    if (!renderer) return;
    cancelAnimationFrame(animationFrame);
    resizeObserver?.disconnect();
    renderer.domElement.removeEventListener('pointerdown', onPovPointerDown);
    window.removeEventListener('pointermove', onPovPointerMove);
    window.removeEventListener('pointerup', onPovPointerUp);
    controls.dispose();
    voxelMesh?.geometry.dispose();
    scene.traverse(node => {
      node.geometry?.dispose?.();
      node.material?.map?.dispose?.();
      node.material?.dispose?.();
    });
    window.__eggGraph?.resume();
    scene = camera = renderer = controls = resizeObserver = undefined;
    voxelMesh = rangeWireframe = null;
  }

  window.addEventListener('egg:occupancy-data', event => applyOccupancy(event.detail));
  window.addEventListener('egg:occupancy-reset', () => {
    if (!camera) return;
    if (povLocked) { povYaw = 0; povPitch = 0; applyPovLook(); return; }
    camera.position.set(DEFAULT_CAMERA_POSITION.x, DEFAULT_CAMERA_POSITION.y, DEFAULT_CAMERA_POSITION.z);
    controls.target.set(DEFAULT_CONTROLS_TARGET.x, DEFAULT_CONTROLS_TARGET.y, DEFAULT_CONTROLS_TARGET.z);
    controls.update();
  });
  window.addEventListener('egg:occupancy-pov-toggle', () => setPovLocked(!povLocked));
  window.addEventListener('egg:occupancy-voxel-scale', () => {
    if (!scene || !lastPayload) return;
    const nativeSize = lastPayload.voxel_size_meters || 0.1;
    const cellSize = nativeSize * voxelScaleMultiplier;
    renderVoxels(rebucketVoxels(lastPayload.voxels || [], cellSize, nativeSize), cellSize);
  });
  window.addEventListener('egg:vision-activate', activate);
  window.addEventListener('egg:vision-deactivate', deactivate);

  if (document.querySelector('.page[data-page="/vision"]')?.classList.contains('active')) activate();
}

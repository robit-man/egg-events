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

function cameraRingRadiusFor(maxRange) {
  return Math.max(0.6, Math.min(3.0, (maxRange || 6.0) * 0.32));
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

  let voxels = [], cameras = [];
  let pixelRatio = 1, width = 1, height = 1;
  let zoom = 60, panX = 0, panY = 0, yaw = 0.5, pitch = -0.25;
  let target = { x: 0, y: 0, z: 0 };
  let drag = null, framed = false;
  const cameraImages = new Map(); // cameraId -> { img, lastLoad }
  const CAMERA_TEXTURE_REFRESH_MS = 3000;

  function point(p) {
    const x = p.x - target.x, y = p.y - target.y, z = p.z - target.z;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const rx = cy * x + sy * z, rz = -sy * x + cy * z, ry = cp * y - sp * rz, depth = sp * y + cp * rz;
    const perspective = Math.max(0.25, Math.min(2.4, 1 - depth / 12));
    return { x: width / 2 + panX + rx * zoom * perspective, y: height / 2 + panY - ry * zoom * perspective, depth, scale: perspective };
  }

  function refreshCameraImage(cameraId) {
    let entry = cameraImages.get(cameraId);
    const now = Date.now();
    if (entry && now - entry.lastLoad < CAMERA_TEXTURE_REFRESH_MS) return entry;
    const img = entry ? entry.img : new Image();
    img.src = `/api/cameras/${encodeURIComponent(cameraId)}/raw.jpg?t=${now}`;
    entry = { img, lastLoad: now };
    cameraImages.set(cameraId, entry);
    return entry;
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
    drag = { x: event.clientX, y: event.clientY, panX, panY, yaw, pitch, pan: event.shiftKey || event.button !== 0 };
  });
  canvas.addEventListener('pointermove', event => {
    canvas.style.cursor = 'grab';
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (drag.pan) { panX = drag.panX + dx; panY = drag.panY + dy; }
    else { yaw = drag.yaw - dx * 0.007; pitch = Math.max(-1.35, Math.min(1.35, drag.pitch + dy * 0.007)); }
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointercancel', () => { drag = null; });
  canvas.addEventListener('contextmenu', event => event.preventDefault());
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoom = Math.max(8, Math.min(400, zoom * Math.exp(-event.deltaY * 0.001)));
  }, { passive: false });

  function applyPayload(payload) {
    if (!payload || !payload.enabled) { voxels = []; cameras = []; return; }
    voxels = payload.voxels || [];
    const radius = cameraRingRadiusFor(payload.max_range_meters);
    cameras = Object.entries(payload.cameras || {}).map(([id, info]) => {
      const yawRad = (Number(info.yaw_degrees || 0) * Math.PI) / 180;
      return {
        id,
        fresh: Number(info.age_seconds ?? 999) < 30,
        x: Math.sin(yawRad) * radius, y: 0.1, z: Math.cos(yawRad) * radius,
      };
    });
    if (!framed && voxels.length) {
      const n = voxels.length;
      target = {
        x: voxels.reduce((sum, v) => sum + v.x, 0) / n,
        y: voxels.reduce((sum, v) => sum + v.y, 0) / n,
        z: voxels.reduce((sum, v) => sum + v.z, 0) / n,
      };
      framed = true;
    }
  }

  function render() {
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = '#070d19';
    context.fillRect(0, 0, width, height);

    const drawables = [];
    for (const v of voxels) drawables.push({ kind: 'voxel', v, ...point(v) });
    for (const cam of cameras) drawables.push({ kind: 'camera', cam, entry: refreshCameraImage(cam.id), ...point(cam) });
    drawables.sort((a, b) => b.depth - a.depth); // paint far first

    for (const d of drawables) {
      if (d.kind === 'voxel') {
        const size = Math.max(1, 3.2 * d.scale);
        const t = Math.max(0, Math.min(1, d.v.confidence ?? 0.5));
        const r = Math.round(0x25 + (0xff - 0x25) * t), g = Math.round(0x63 + (0xae - 0x63) * t), b = Math.round(0xeb + (0x00 - 0xeb) * t);
        context.fillStyle = `rgb(${r},${g},${b})`;
        context.fillRect(d.x - size / 2, d.y - size / 2, size, size);
      } else {
        const img = d.entry.img;
        const h = 70 * d.scale;
        const w = img.naturalWidth && img.naturalHeight ? h * (img.naturalWidth / img.naturalHeight) : h;
        if (img.complete && img.naturalWidth) {
          context.globalAlpha = d.cam.fresh ? 0.95 : 0.5;
          context.drawImage(img, d.x - w / 2, d.y - h / 2, w, h);
          context.globalAlpha = 1;
        } else {
          context.fillStyle = '#223344';
          context.fillRect(d.x - w / 2, d.y - h / 2, w, h);
        }
        context.strokeStyle = d.cam.fresh ? '#34d399' : '#667085';
        context.lineWidth = 1;
        context.strokeRect(d.x - w / 2, d.y - h / 2, w, h);
      }
    }
    requestAnimationFrame(render);
  }

  window.addEventListener('egg:occupancy-data', event => applyPayload(event.detail));
  window.addEventListener('egg:occupancy-reset', () => {
    target = { x: 0, y: 0, z: 0 }; yaw = 0.5; pitch = -0.25; zoom = 60; panX = 0; panY = 0; framed = false;
  });
  window.dispatchEvent(new CustomEvent('egg:occupancy-renderer', { detail: { mode: 'canvas-2d' } }));
  render();
}

const container = document.getElementById('occupancy-scene');

if (container) {
  let scene, camera, renderer, controls, resizeObserver, animationFrame;
  let voxelMesh = null, cameraRoot = null, rangeWireframe = null;
  let lastPayload = null;

  const voxelGeometry = new THREE.BoxGeometry(1, 1, 1);
  const voxelMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05 });
  const lowColor = new THREE.Color(0x2563eb);
  const highColor = new THREE.Color(0xffae00);
  const tmpColor = new THREE.Color();
  const tmpMatrix = new THREE.Matrix4();

  // Each contributing camera's live frame, textured onto a plane and
  // placed radially at that camera's known array yaw -- same rotation
  // convention core/occupancy.py uses to fuse depth into the shared
  // frame (yaw about +Y, 0deg = +Z), so a camera's image sits in the
  // scene exactly where the voxels it contributed radiate outward from.
  const textureLoader = new THREE.TextureLoader();
  const cameraPlanes = new Map(); // cameraId -> { mesh, lastTextureLoad }
  const CAMERA_TEXTURE_REFRESH_MS = 3000;
  const CAMERA_PLANE_HEIGHT = 0.55;

  function cameraRingRadius(maxRange) {
    return THREE.MathUtils.clamp((maxRange || 6.0) * 0.32, 0.6, 3.0);
  }

  function renderVoxels(voxels, voxelSize) {
    if (voxelMesh) { scene.remove(voxelMesh); voxelMesh.geometry.dispose(); voxelMesh = null; }
    if (!voxels.length) return;
    const size = Math.max(voxelSize, 0.01);
    voxelMesh = new THREE.InstancedMesh(voxelGeometry, voxelMaterial, voxels.length);
    voxelMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(voxels.length * 3), 3);
    voxels.forEach((voxel, index) => {
      tmpMatrix.makeScale(size * 0.92, size * 0.92, size * 0.92);
      tmpMatrix.setPosition(voxel.x, voxel.y, voxel.z);
      voxelMesh.setMatrixAt(index, tmpMatrix);
      tmpColor.copy(lowColor).lerp(highColor, THREE.MathUtils.clamp(voxel.confidence ?? 0.5, 0, 1));
      voxelMesh.setColorAt(index, tmpColor);
    });
    voxelMesh.instanceMatrix.needsUpdate = true;
    if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
    scene.add(voxelMesh);
  }

  function disposeCameraPlane(entry) {
    entry.mesh.geometry.dispose();
    entry.mesh.material.map?.dispose();
    entry.mesh.material.dispose();
  }

  function refreshCameraTexture(cameraId, mesh) {
    const url = `/api/cameras/${encodeURIComponent(cameraId)}/raw.jpg?t=${Date.now()}`;
    textureLoader.load(
      url,
      texture => {
        texture.colorSpace = THREE.SRGBColorSpace;
        const image = texture.image;
        if (image && image.width && image.height) {
          const aspect = image.width / image.height;
          mesh.geometry.dispose();
          mesh.geometry = new THREE.PlaneGeometry(CAMERA_PLANE_HEIGHT * aspect, CAMERA_PLANE_HEIGHT);
        }
        mesh.material.map?.dispose();
        mesh.material.map = texture;
        mesh.material.color.set(0xffffff);
        mesh.material.needsUpdate = true;
      },
      undefined,
      () => {}, // camera frame not available yet -- keep the placeholder tile
    );
  }

  // Each contributing camera's live frame, textured onto a plane
  // positioned radially at that camera's known array yaw and facing the
  // shared origin -- matches core/occupancy.py's rotation convention
  // (yaw about +Y, +Z is the video0-local boresight), so a camera's
  // image sits exactly where the voxels it contributed radiate from.
  function renderCameraMarkers(cameras, maxRange) {
    const radius = cameraRingRadius(maxRange);
    const seen = new Set();
    Object.entries(cameras || {}).forEach(([cameraId, info]) => {
      seen.add(cameraId);
      const yaw = THREE.MathUtils.degToRad(Number(info.yaw_degrees || 0));
      const x = Math.sin(yaw) * radius;
      const z = Math.cos(yaw) * radius;
      let entry = cameraPlanes.get(cameraId);
      if (!entry) {
        const material = new THREE.MeshBasicMaterial({
          color: 0x223344, side: THREE.DoubleSide, transparent: true, opacity: 0.96,
        });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(CAMERA_PLANE_HEIGHT, CAMERA_PLANE_HEIGHT), material);
        mesh.userData.cameraId = cameraId;
        cameraRoot.add(mesh);
        entry = { mesh, lastTextureLoad: 0 };
        cameraPlanes.set(cameraId, entry);
      }
      entry.mesh.position.set(x, 0.05, z);
      entry.mesh.lookAt(0, 0.05, 0);
      const fresh = Number(info.age_seconds ?? 999) < 30;
      entry.mesh.material.opacity = fresh ? 0.96 : 0.5;
      const now = Date.now();
      if (now - entry.lastTextureLoad > CAMERA_TEXTURE_REFRESH_MS) {
        entry.lastTextureLoad = now;
        refreshCameraTexture(cameraId, entry.mesh);
      }
    });
    for (const [cameraId, entry] of cameraPlanes) {
      if (seen.has(cameraId)) continue;
      cameraRoot.remove(entry.mesh);
      disposeCameraPlane(entry);
      cameraPlanes.delete(cameraId);
    }
  }

  function clearCameraMarkers() {
    for (const entry of cameraPlanes.values()) {
      cameraRoot.remove(entry.mesh);
      disposeCameraPlane(entry);
    }
    cameraPlanes.clear();
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
    const radius = cameraRingRadius(payload.max_range_meters);
    const box = new THREE.Box3();
    box.expandByPoint(new THREE.Vector3(-radius, 0, -radius));
    box.expandByPoint(new THREE.Vector3(radius, 0.5, radius));
    for (const v of voxels) box.expandByPoint(new THREE.Vector3(v.x, v.y, v.z));
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
      clearCameraMarkers();
      return;
    }
    renderVoxels(payload.voxels || [], payload.voxel_size_meters || 0.1);
    renderCameraMarkers(payload.cameras || {}, payload.max_range_meters);
    renderRangeWireframe(payload.max_range_meters);
    if (!framedOnce) fitCameraToScene(payload);
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
    controls.update();
    renderer.render(scene, camera);
  }

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
    camera.position.set(3.2, 2.4, 4.2);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 0.4;
    controls.maxDistance = 60;
    controls.target.set(0, 0.3, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 0.6);
    key.position.set(3, 6, 4);
    scene.add(key);
    scene.add(new THREE.AxesHelper(0.35));

    cameraRoot = new THREE.Group();
    scene.add(cameraRoot);

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();
    animate();

    if (lastPayload) applyOccupancy(lastPayload);
  }

  function deactivate() {
    if (!renderer) return;
    cancelAnimationFrame(animationFrame);
    resizeObserver?.disconnect();
    controls.dispose();
    voxelMesh?.geometry.dispose();
    scene.traverse(node => {
      node.geometry?.dispose?.();
      node.material?.map?.dispose?.();
      node.material?.dispose?.();
    });
    cameraPlanes.clear(); // meshes just disposed above belong to the now-discarded scene
    window.__eggGraph?.resume();
    scene = camera = renderer = controls = resizeObserver = undefined;
    voxelMesh = cameraRoot = rangeWireframe = null;
  }

  window.addEventListener('egg:occupancy-data', event => applyOccupancy(event.detail));
  window.addEventListener('egg:occupancy-reset', () => {
    if (!camera) return;
    camera.position.set(3.2, 2.4, 4.2);
    controls.target.set(0, 0.3, 0);
    controls.update();
  });
  window.addEventListener('egg:vision-activate', activate);
  window.addEventListener('egg:vision-deactivate', deactivate);

  if (document.querySelector('.page[data-page="/vision"]')?.classList.contains('active')) activate();
}

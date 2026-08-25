import * as THREE from 'three';
import { OrbitControls } from '/assets/OrbitControls.js?v=20260811a';

// Renders the fused voxel occupancy grid published by /api/occupancy --
// one shared "egg frame" (Y up, +Z the video0-camera-array boresight, see
// core/occupancy.py's module docstring) built by rotating each camera's
// monocular depth by its known array yaw before back-projecting.

function initUnavailable(container, message) {
  const note = document.createElement('div');
  note.className = 'empty';
  note.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;';
  note.textContent = message;
  container.appendChild(note);
}

const container = document.getElementById('occupancy-scene');

if (container) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    const context = renderer.getContext();
    const debug = context.getExtension('WEBGL_debug_renderer_info');
    const implementation = String(debug ? context.getParameter(debug.UNMASKED_RENDERER_WEBGL) : '');
    if (/swiftshader|llvmpipe|software/i.test(implementation)) {
      renderer.dispose();
      renderer = undefined;
      initUnavailable(container, '3D voxel view needs real WebGL acceleration, unavailable in this browser context.');
    }
  } catch (error) {
    initUnavailable(container, '3D voxel view unavailable: WebGL failed to initialize.');
  }

  if (renderer) {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x070d19, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    camera.position.set(3.2, 2.4, 4.2);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 0.4;
    controls.maxDistance = 60;
    controls.target.set(0, 0.3, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 0.6);
    key.position.set(3, 6, 4);
    scene.add(key);

    const originMarker = new THREE.AxesHelper(0.35);
    scene.add(originMarker);

    const voxelRoot = new THREE.Group();
    scene.add(voxelRoot);
    const cameraRoot = new THREE.Group();
    scene.add(cameraRoot);
    let rangeWireframe = null;

    const voxelGeometry = new THREE.BoxGeometry(1, 1, 1);
    const voxelMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05 });
    let voxelMesh = null;
    const lowColor = new THREE.Color(0x2563eb);
    const highColor = new THREE.Color(0xffae00);
    const tmpColor = new THREE.Color();
    const tmpMatrix = new THREE.Matrix4();

    function clearVoxelMesh() {
      if (!voxelMesh) return;
      voxelRoot.remove(voxelMesh);
      voxelMesh.geometry.dispose();
      voxelMesh = null;
    }

    function renderVoxels(voxels, voxelSize) {
      clearVoxelMesh();
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
      voxelRoot.add(voxelMesh);
    }

    function clearCameraMarkers() {
      while (cameraRoot.children.length) {
        const child = cameraRoot.children.pop();
        child.geometry?.dispose?.();
        child.material?.dispose?.();
      }
    }

    // Small cone at the shared origin per contributing camera, pointed
    // along that camera's known array yaw -- matches core/occupancy.py's
    // rotation convention (yaw about +Y, +Z is the video0-local boresight).
    function renderCameraMarkers(cameras) {
      clearCameraMarkers();
      const coneGeometry = new THREE.ConeGeometry(0.05, 0.16, 12);
      Object.entries(cameras || {}).forEach(([cameraId, info]) => {
        const yaw = THREE.MathUtils.degToRad(Number(info.yaw_degrees || 0));
        const fresh = Number(info.age_seconds ?? 999) < 30;
        const material = new THREE.MeshStandardMaterial({ color: fresh ? 0x34d399 : 0x667085 });
        const marker = new THREE.Mesh(coneGeometry, material);
        marker.rotation.x = Math.PI / 2;
        marker.position.set(Math.sin(yaw) * 0.22, 0.05, Math.cos(yaw) * 0.22);
        marker.rotation.z = -yaw;
        marker.userData.cameraId = cameraId;
        cameraRoot.add(marker);
      });
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

    function applyOccupancy(payload) {
      if (!payload || !payload.enabled) {
        clearVoxelMesh();
        clearCameraMarkers();
        return;
      }
      renderVoxels(payload.voxels || [], payload.voxel_size_meters || 0.1);
      renderCameraMarkers(payload.cameras || {});
      renderRangeWireframe(payload.max_range_meters);
    }

    window.addEventListener('egg:occupancy-data', event => applyOccupancy(event.detail));
    window.addEventListener('egg:occupancy-reset', () => {
      camera.position.set(3.2, 2.4, 4.2);
      controls.target.set(0, 0.3, 0);
      controls.update();
    });

    function resize() {
      const width = container.clientWidth || 1;
      const height = container.clientHeight || 1;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    }
    new ResizeObserver(resize).observe(container);
    resize();

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  }
}

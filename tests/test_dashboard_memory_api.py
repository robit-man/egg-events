import asyncio
from types import SimpleNamespace

from egg_companion.config import EggConfig
from egg_companion.services import dashboard
from egg_companion.services.audit import AuditCheck


def test_readiness_monitor_replaces_stale_failure_after_recovery() -> None:
    async def scenario() -> None:
        responses = [
            [AuditCheck("omnius-cognition", "fail", "timeout")],
            [AuditCheck("omnius-cognition", "pass", "READY")],
        ]

        async def probe(_):
            return responses.pop(0)

        monitor = dashboard.ReadinessMonitor(
            SimpleNamespace(),
            probe=probe,
            healthy_interval_seconds=300,
            degraded_interval_seconds=0,
        )
        assert await monitor.poll() == []
        await asyncio.sleep(0)
        assert (await monitor.poll())[0].status == "fail"
        await asyncio.sleep(0)
        recovered = await monitor.poll()

        assert recovered == [AuditCheck("omnius-cognition", "pass", "READY")]
        assert monitor.snapshot()["updated_at"] is not None

    asyncio.run(scenario())


def test_dashboard_registers_governance_routes_and_audit_does_not_block(monkeypatch) -> None:
    async def scenario() -> None:
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default", "doa_mode": "disabled"},
                "omnius": {"model": "test", "voice_model": "test"},
                "identity": {"enabled": False},
                "object_learning": {"enabled": False},
                "memory": {"enabled": False},
                "camera_discovery": {"enabled": False},
            }
        )
        created: list[object] = []
        started = asyncio.Event()

        class FakeRuntime:
            def __init__(self, runtime_config) -> None:
                created.append(self)
                self.config = runtime_config

            @staticmethod
            async def run() -> None:
                await asyncio.Event().wait()

        class FakeRunner:
            def __init__(self, app, **kwargs) -> None:
                self.app = app
                captured.app = app

            async def setup(self) -> None:
                return None

            async def cleanup(self) -> None:
                return None

        class FakeSite:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def start(self) -> None:
                started.set()

        async def failed_audit(_):
            return [AuditCheck("cuda", "fail", "diagnostic failure")]

        captured = SimpleNamespace(app=None)
        monkeypatch.setattr(dashboard, "CompanionRuntime", FakeRuntime)
        monkeypatch.setattr(dashboard, "audit_hardware", failed_audit)
        monkeypatch.setattr(dashboard.web, "AppRunner", FakeRunner)
        monkeypatch.setattr(dashboard.web, "TCPSite", FakeSite)

        task = asyncio.create_task(dashboard.serve_dashboard(config, 0))
        try:
            waiter = asyncio.create_task(started.wait())
            done, _ = await asyncio.wait(
                {task, waiter}, timeout=2, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                task.result()
            assert waiter in done, "dashboard site did not start"
            paths = {route.resource.canonical for route in captured.app.router.routes()}
            assert created, "runtime must start even when an audit diagnostic fails"
            assert {
                "/api/config",
                "/api/graph",
                "/api/graph/node",
                "/api/voice/action",
                "/api/chat/message",
                "/api/actions/focus_camera",
                "/api/actions/inspect_entity",
                "/api/voice/conversation",
                "/api/dreams",
                "/api/dreams/run",
                "/api/memory/narratives",
                "/api/memory/narratives/{local_date}",
                "/api/identities/{profile_id}/timeline",
                "/api/identities/{profile_id}/samples/{sample_id}.jpg",
                "/api/memory/episodes",
                "/api/memory/entities/{entity_id}",
                "/api/memory/evidence/{evidence_id}/media",
                "/api/memory/claims",
                "/api/memory/revisions",
                "/api/memory/export/{entity_id}",
                "/api/cognition/state",
                "/api/occupancy",
                "/api/occupancy/resolution",
            } <= paths
            assert "/assets" in paths
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_dashboard_application_is_professional_spa_with_local_graph_assets() -> None:
    assert "EGG / COMPANION" not in dashboard.PAGE
    assert "live sensory field · associative memory · local cognition" not in dashboard.PAGE
    assert "Optical Array · Raw Streams + Instance Masks" not in dashboard.PAGE
    assert 'data-page="/graph"' in dashboard.PAGE
    assert 'href="/dreams" data-route="/dreams"' in dashboard.PAGE
    assert 'data-page="/dreams"' in dashboard.PAGE
    assert 'href="/narrative" data-route="/narrative"' in dashboard.PAGE
    assert 'data-page="/narrative"' in dashboard.PAGE
    assert "Latest chronological replay" in dashboard.PAGE
    assert "loadNarrativeDetail" in dashboard.PAGE
    assert "narrative-artifacts" in dashboard.PAGE
    assert 'href="/configuration" data-route="/configuration"' in dashboard.PAGE
    assert 'src="/assets/knowledge_graph.js?v=20260824e"' in dashboard.PAGE
    assert '"three":"/assets/three.module.min.js"' in dashboard.PAGE
    assert "window.open(" not in dashboard.PAGE
    assert "graphDataSignature" in dashboard.PAGE
    assert "egg:graph-activations" in dashboard.PAGE
    assert "graphActivationSequence" in dashboard.PAGE
    assert "Heard voice" in dashboard.PAGE
    assert "Memory recall" in dashboard.PAGE
    assert "loadGraph(true), 2000" in dashboard.PAGE
    assert "Connected evidence and artifacts" in dashboard.PAGE
    assert 'id="graph-theater"' in dashboard.PAGE
    assert 'id="graph-fullscreen"' in dashboard.PAGE
    assert 'class="graph-detail-grid"' in dashboard.PAGE
    assert ".graph-panel:fullscreen" in dashboard.PAGE
    assert "egg.graph.theater" in dashboard.PAGE
    assert "Spline form is evidence, not decoration" in dashboard.PAGE
    assert "Applied in-page; an Egg restart is not required" in dashboard.PAGE
    assert "voice-service-state" in dashboard.PAGE
    assert "voiceFormDirty" in dashboard.PAGE
    assert "Complete durable audible ledger" in dashboard.PAGE
    assert "conversationLedger" in dashboard.PAGE
    assert "/api/voice/conversation?limit=5000" in dashboard.PAGE
    assert "message-tags" in dashboard.PAGE
    assert "turn.tags" in dashboard.PAGE
    assert "health recheck running" in dashboard.PAGE
    assert "Cognition unavailable" in dashboard.PAGE
    assert "Audio comprehension unavailable" in dashboard.PAGE
    assert 'href="/chat" data-route="/chat"' in dashboard.PAGE
    assert 'data-page="/chat"' in dashboard.PAGE
    assert 'id="chat-form"' in dashboard.PAGE
    assert 'id="chat-conversation"' in dashboard.PAGE
    assert 'id="chat-input"' in dashboard.PAGE
    assert "/api/chat/message" in dashboard.PAGE
    assert "renderConversation(telemetry, '#chat-conversation', true)" in dashboard.PAGE
    assert "border-radius: 0 !important" in dashboard.PAGE
    assert 'data-person-id=' in dashboard.PAGE
    assert 'id="person-inspector"' in dashboard.PAGE
    assert "loadPersonTimeline" in dashboard.PAGE
    assert "encounter periods" in dashboard.PAGE
    assert '<option value="person">People</option>' in dashboard.PAGE
    assert '<option value="daily_narrative">Daily stories</option>' in dashboard.PAGE
    assert 'data-graph-kind="world_model"' in dashboard.PAGE
    assert "Hover to isolate · click to lock the filter" in dashboard.PAGE
    assert 'id="occupancy-scene"' in dashboard.PAGE
    assert 'src="/assets/occupancy_scene.js?v=20260825d"' in dashboard.PAGE
    assert "egg:occupancy-data" in dashboard.PAGE
    assert "loadOccupancy" in dashboard.PAGE
    assert 'id="occupancy-voxel-scale-up"' in dashboard.PAGE
    assert 'id="occupancy-voxel-scale-down"' in dashboard.PAGE
    assert "egg:occupancy-voxel-scale" in dashboard.PAGE


def test_occupancy_scene_borrows_graphs_webgl_renderer_instead_of_opening_a_second() -> None:
    """Confirmed on real hardware (Chromium GPU-process log: "Could not
    create a WebGL context ... GL_VENDOR = Disabled, Sandboxed = yes,
    BindToCurrentSequence failed") that this browser/GPU can only sustain
    ONE live WebGL context -- a second, fully independent
    THREE.WebGLRenderer fails outright while knowledge_graph.js's is
    alive. occupancy_scene.js must borrow that exact renderer via
    window.__eggGraph rather than ever constructing its own."""
    graph_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "knowledge_graph.js"
    ).read_text()
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "window.__eggGraph =" in graph_source
    assert "pause()" in graph_source
    assert "resume()" in graph_source
    assert "new THREE.WebGLRenderer" not in occupancy_source
    assert "window.__eggGraph" in occupancy_source
    assert "shared.pause()" in occupancy_source
    assert "window.__eggGraph?.resume()" in occupancy_source
    assert "egg:vision-activate" in occupancy_source
    assert "egg:vision-deactivate" in occupancy_source


def test_occupancy_scene_has_no_camera_sprite_previews() -> None:
    """Camera source previews (textured planes / drawn images at each
    camera's array position) were deliberately removed from both render
    paths -- the 3D view shows only the fused voxel reconstruction now."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "/api/cameras/" not in occupancy_source
    assert "raw.jpg" not in occupancy_source
    assert "TextureLoader" not in occupancy_source
    assert "renderCameraMarkers" not in occupancy_source
    assert "refreshCameraImage" not in occupancy_source


def test_occupancy_scene_pov_lock_looks_around_from_the_fused_frame_origin() -> None:
    """The Egg POV toolbar button locks the camera to (0,0,0) in the
    fused frame -- per core/occupancy.py's module docstring, that origin
    IS the rig's single modeled optical center (physically the center of
    the 256mm ring the four cameras mount around) -- and dragging looks
    around from there (FPS-style) instead of orbiting a target. Must be
    implemented in both the WebGL and 2D-canvas-fallback render paths."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "egg:occupancy-pov-toggle" in occupancy_source
    assert "egg:occupancy-pov-changed" in occupancy_source
    assert occupancy_source.count("povLocked") > 4  # present in both render paths
    assert "camera.position.set(0, 0, 0)" in occupancy_source  # WebGL path
    assert "function povPoint(p)" in occupancy_source  # 2D fallback path
    assert "controls.enabled = !locked" in occupancy_source


def test_occupancy_scene_colors_voxels_from_source_frame_not_confidence_gradient() -> None:
    """Voxels must be colored by the real RGB sampled from the source
    camera frame at the pixel that produced them (core/occupancy.py's
    color_frame param), in both render paths (WebGL InstancedMesh and the
    2D canvas fallback) -- not a synthetic confidence-lerp gradient."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "voxel.color" in occupancy_source
    assert "d.v.color" in occupancy_source
    assert "lowColor" not in occupancy_source
    assert "highColor" not in occupancy_source


def test_occupancy_scene_voxel_size_scales_with_zoom_and_manual_control() -> None:
    """Voxel visual size was previously a fixed pixel constant, so it
    stayed static as the user scrolled to zoom. Voxels now render as real
    projected cubes: each face corner is a real-world offset
    (voxelWorldSize = voxelSizeMeters * voxelScaleMultiplier) that goes
    through the same point() perspective/zoom projection as every other
    scene point, so on-screen size follows actual spatial geometry
    (distance, zoom, viewing angle) rather than an arbitrary flat pixel
    constant -- and a toolbar Voxel +/- control applies an additional
    manual multiplier, honored in both render paths."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "voxelWorldSize = voxelSizeMeters * voxelScaleMultiplier" in occupancy_source
    assert "lx * voxelWorldSize" in occupancy_source
    assert "egg:occupancy-voxel-scale" in occupancy_source
    assert "voxelScaleMultiplier" in occupancy_source
    assert "* voxelScaleMultiplier" in occupancy_source  # applied in the WebGL path too


def test_occupancy_scene_voxels_render_as_real_projected_cubes() -> None:
    """Voxels must be real projected cube geometry (6 possible faces,
    only the camera-facing subset drawn each frame via backface culling
    on the rotated face normal), not a flat screen-aligned square sprite."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "CUBE_FACES" in occupancy_source
    assert "visibleCubeFaces" in occupancy_source
    assert "face.corners.map" in occupancy_source
    assert "context.closePath()" in occupancy_source
    assert "context.fillRect(d.x - size / 2" not in occupancy_source  # old flat-sprite path removed


def test_occupancy_scene_camera_array_reads_left_to_right_on_screen() -> None:
    """Screen X is mirrored consistently in both render paths (voxels
    project through the same convention in both) so the panoramic array
    reads left-to-right on screen, and the orbit-drag yaw direction is
    compensated to still feel natural after that mirror -- the backend
    fusion geometry in core/occupancy.py is untouched, this is display-only."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "panX - rx * zoom * perspective" in occupancy_source
    assert "tmpMatrix.setPosition(-voxel.x, voxel.y, voxel.z)" in occupancy_source
    assert "yaw = drag.yaw + dx * 0.007" in occupancy_source


def test_occupancy_scene_increasing_voxel_size_rebuckets_instead_of_producing_gaps() -> None:
    """Increasing the Voxel +/- control must merge neighboring native
    voxels into larger, gap-filling blocks (client-side rebucketing at
    the coarser effective cell size), not just inflate each native
    voxel's own footprint in place -- which only produces overlap when
    enlarged and visible negative space between unmoved points when
    shrunk. Both render paths (WebGL and the 2D canvas fallback) must
    use the shared rebucketVoxels() helper rather than rendering the raw
    per-voxel list directly at a scaled size."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "function rebucketVoxels(rawVoxels, cellSize, nativeSize)" in occupancy_source
    assert "Math.floor(v.x / cellSize)" in occupancy_source
    assert "renderVoxels(rebucketVoxels(payload.voxels || [], cellSize, nativeSize), cellSize)" in occupancy_source
    assert "voxels = rebucketVoxels(rawVoxels, voxelSizeMeters * voxelScaleMultiplier, voxelSizeMeters)" in occupancy_source


def test_occupancy_scene_resolution_control_exists_next_to_voxel_size() -> None:
    """A separate Resolution +/- control (backend sample_stride, how many
    of DA3's actual per-frame depth points get processed into voxels)
    must exist next to the client-side Voxel +/- display-scale control,
    PUTting to /api/occupancy/resolution rather than only adjusting the
    already-fetched voxel list."""
    assert 'id="occupancy-resolution-up"' in dashboard.PAGE
    assert 'id="occupancy-resolution-down"' in dashboard.PAGE
    assert "/api/occupancy/resolution" in dashboard.PAGE
    assert "adjustOccupancyResolution" in dashboard.PAGE
    assert "currentSampleStride" in dashboard.PAGE


def test_occupancy_pov_lock_button_exists_and_updates_hint_text() -> None:
    """The Egg POV toolbar button must dispatch the toggle event the
    renderer listens for, and the dashboard must update the hint text /
    pressed state in response to the renderer's own change event -- not
    just fire-and-forget the click."""
    assert 'id="occupancy-pov-lock"' in dashboard.PAGE
    assert "egg:occupancy-pov-toggle" in dashboard.PAGE
    assert "egg:occupancy-pov-changed" in dashboard.PAGE
    assert 'id="occupancy-hint"' in dashboard.PAGE


def test_occupancy_scene_auto_frames_camera_on_real_voxel_data() -> None:
    """The camera must frame itself on the actual returned voxel bounds
    once real data arrives, rather than trusting one hardcoded position
    to suit every room the array is capturing."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "fitCameraToScene" in occupancy_source
    assert "Box3" in occupancy_source
    assert "framedOnce" in occupancy_source


def test_occupancy_scene_falls_back_to_2d_canvas_when_no_webgl_exists_anywhere() -> None:
    """Confirmed on real hardware (browser console log): knowledge_graph.js's
    own WebGLRenderer construction fails identically to occupancy_scene.js's
    borrowed-renderer attempt ("GL_VENDOR = Disabled ... BindToCurrentSequence
    failed") because this Chromium instance's GPU process is launched with
    --use-gl=disabled -- a real, deterministic condition (likely a
    GPU-less remote-desktop/X11 session), not a transient failure. When no
    live WebGL context exists anywhere on the page, occupancy_scene.js
    must render via a pure 2D canvas (mirroring knowledge_graph.js's own
    proven initCanvasFallback technique) rather than leaving the /vision
    page permanently inert."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "function initCanvasFallback(container)" in occupancy_source
    assert "getContext('2d'" in occupancy_source
    assert "fallbackActive = true" in occupancy_source
    assert "initCanvasFallback(container)" in occupancy_source
    # Same rotation/projection technique as knowledge_graph.js's own fallback.
    assert "Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch)" in occupancy_source


def test_graph_horizontal_orbit_is_flipped_in_webgl_and_canvas_renderers() -> None:
    graph_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "knowledge_graph.js"
    ).read_text()
    controls_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "OrbitControls.js"
    ).read_text()

    assert "yaw=drag.yaw-dx*.007" in graph_source
    assert "OrbitControls.js?v=20260811a" in graph_source
    assert "_rotateLeft( - _twoPI * this._rotateDelta.x" in controls_source
    assert "egg:graph-activations" in graph_source
    assert "activationHopMs" in graph_source
    assert "activePulseObjects" in graph_source
    assert "associativeLayout3D" in graph_source
    assert "ASSOCIATIVE_RELATIONS" in graph_source
    assert "Math.hypot(dx, dy, dz)" in graph_source
    assert "global depth plane" in graph_source
    assert "structuralAgreement" in graph_source
    assert "communityLabels" in graph_source
    assert "volumePoint" in graph_source
    assert "radialSeed" not in graph_source
    assert "RELATION_GEOMETRY" in graph_source
    assert "associativeSplinePoints" in graph_source
    assert "associationStrength" in graph_source
    assert "confirmations" in graph_source
    assert "thickness" in graph_source
    assert "arch" in graph_source
    assert "angle" in graph_source
    assert "new THREE.CatmullRomCurve3" in graph_source
    assert "graphNodeModality" in graph_source
    assert "graphNodeMatches" in graph_source
    assert "controls.minDistance = 1.2" in graph_source
    assert "controls.zoomToCursor = false" in graph_source
    assert "Math.min(240,zoom*factor)" in graph_source

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
    assert 'src="/assets/knowledge_graph.js?v=20260824c"' in dashboard.PAGE
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
    assert 'src="/assets/occupancy_scene.js?v=20260824d"' in dashboard.PAGE
    assert "egg:occupancy-data" in dashboard.PAGE
    assert "loadOccupancy" in dashboard.PAGE


def test_occupancy_scene_builds_its_own_renderer_matching_graphs_pattern() -> None:
    """occupancy_scene.js constructs its own THREE.WebGLRenderer using the
    exact same options and software-renderer fallback check as
    knowledge_graph.js's proven, known-working setup, and fully releases
    the GPU context (dispose + forceContextLoss) when /vision goes
    inactive rather than leaving it live for the whole page lifetime."""
    graph_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "knowledge_graph.js"
    ).read_text()
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "window.__eggGraph =" not in graph_source
    assert "new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })" in graph_source
    assert "new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })" in occupancy_source
    assert "swiftshader|llvmpipe|software" in occupancy_source
    assert "renderer.dispose()" in occupancy_source
    assert "renderer.forceContextLoss()" in occupancy_source
    assert "egg:vision-activate" in occupancy_source
    assert "egg:vision-deactivate" in occupancy_source


def test_occupancy_scene_places_camera_feeds_radially_matching_fusion_yaw() -> None:
    """Each contributing camera's live frame must be textured onto a
    plane positioned using the exact same yaw-rotation convention
    core/occupancy.py uses to fuse depth into the shared frame (yaw
    about +Y, 0deg = +Z: x=sin(yaw)*r, z=cos(yaw)*r) -- otherwise the
    camera imagery and the voxels it produced would visually disagree."""
    occupancy_source = (
        dashboard.Path(dashboard.__file__).with_name("vendor") / "occupancy_scene.js"
    ).read_text()

    assert "/api/cameras/" in occupancy_source
    assert "raw.jpg" in occupancy_source
    assert "TextureLoader" in occupancy_source
    assert "Math.sin(yaw) * radius" in occupancy_source
    assert "Math.cos(yaw) * radius" in occupancy_source
    assert "lookAt(0, 0.05, 0)" in occupancy_source


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

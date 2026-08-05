import asyncio
from types import SimpleNamespace

from egg_companion.config import EggConfig
from egg_companion.services import dashboard
from egg_companion.services.audit import AuditCheck


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
                "/api/memory/episodes",
                "/api/memory/entities/{entity_id}",
                "/api/memory/claims",
                "/api/memory/revisions",
                "/api/memory/export/{entity_id}",
                "/api/cognition/state",
            } <= paths
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

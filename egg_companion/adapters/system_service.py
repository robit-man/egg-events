from __future__ import annotations

import os

import aiohttp

from egg_companion.config import SystemServiceConfig


class SystemServiceClient:
    def __init__(self, config: SystemServiceConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        if not self.config.bearer_token_env:
            return {}
        token = os.getenv(self.config.bearer_token_env)
        if not token:
            raise RuntimeError(f"required token environment variable is unset: {self.config.bearer_token_env}")
        return {"Authorization": f"Bearer {token}"}

    async def health(self) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
            async with session.get(
                f"{str(self.config.base_url).rstrip('/')}{self.config.status_path}", headers=self._headers()
            ) as response:
                response.raise_for_status()

    async def publish_event(self, event: dict[str, object]) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}{self.config.event_path}",
                json=event,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()

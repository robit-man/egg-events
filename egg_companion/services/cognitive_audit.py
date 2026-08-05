from __future__ import annotations

import json

from egg_companion.config import EggConfig
from egg_companion.memory.store import MemoryStore


def audit_cognitive_memory(config: EggConfig) -> tuple[bool, dict[str, object]]:
    store = MemoryStore(config.memory)
    try:
        report = store.integrity_report()
    finally:
        store.close()
    passed = bool(
        report["sqlite_integrity"] == "ok"
        and report["journal_mode"] == "wal"
        and report["writable"] is True
        and not report["foreign_key_violations"]
        and report["orphan_entity_embeddings"] == 0
        and not report["duplicate_legacy_sources"]
        and not report["media"]["missing_evidence_ids"]
        and not report["media"]["checksum_mismatch_evidence_ids"]
    )
    report["status"] = "pass" if passed else "fail"
    return passed, report


def format_cognitive_audit(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)

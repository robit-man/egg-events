from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from egg_companion.config import load_config
from egg_companion.evaluation import evaluate_trace, load_trace
from egg_companion.memory.migrate_legacy import LegacyMemoryMigrator
from egg_companion.memory.retention import RetentionPlanner
from egg_companion.memory.store import MemoryStore
from egg_companion.runtime import CompanionRuntime
from egg_companion.services.identity import IdentityLibrary
from egg_companion.services.object_library import ObjectLibrary
from egg_companion.services.audit import audit_hardware, format_audit, readiness_passes
from egg_companion.services.dashboard import serve_dashboard
from egg_companion.services.cognitive_audit import audit_cognitive_memory, format_cognitive_audit
from egg_companion.services.live_trace import format_live_trace, trace_live_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egg-companion")
    parser.add_argument("--config", default="config/egg.yaml", help="path to Egg YAML configuration")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("audit", help="probe configured hardware and services")
    subcommands.add_parser("run", help="start the live companion runtime")
    serve = subcommands.add_parser("serve", help="start dashboard and launch companion when ready")
    serve.add_argument("--port", type=int, default=8788)
    subcommands.add_parser("memory-audit", help="verify graph integrity and cognitive-memory invariants")
    trace = subcommands.add_parser("trace", help="record a timed live hardware/runtime trace")
    trace.add_argument("--url", default="http://127.0.0.1:8788", help="running Egg dashboard URL")
    trace.add_argument("--seconds", type=float, default=10.0)
    evaluate = subcommands.add_parser("evaluate", help="score a deterministic offline trace")
    evaluate.add_argument("--trace", required=True, help="path to a metadata-only JSON trace")
    memory = subcommands.add_parser("memory", help="migrate or verify cognitive memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_commands.add_parser("migrate", help="idempotently import legacy profiles")
    memory_commands.add_parser("verify", help="verify DB, media checksums, and retention plan")
    return parser


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    logging.basicConfig(level=config.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "audit":
        checks = await audit_hardware(config)
        print(format_audit(checks))
        return 0 if readiness_passes(checks) else 1
    if args.command == "serve":
        await serve_dashboard(config, args.port)
        return 0
    if args.command == "memory-audit":
        passed, report = await asyncio.to_thread(audit_cognitive_memory, config)
        print(format_cognitive_audit(report))
        return 0 if passed else 1
    if args.command == "trace":
        passed, report = await trace_live_runtime(args.url, args.seconds)
        print(format_live_trace(report))
        return 0 if passed else 1
    if args.command == "evaluate":
        passed, report = evaluate_trace(load_trace(args.trace))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if passed else 1
    if args.command == "memory":
        store = MemoryStore(config.memory)
        try:
            if args.memory_command == "migrate":
                report = LegacyMemoryMigrator(
                    store, IdentityLibrary(config.identity), ObjectLibrary(config.object_learning)
                ).run()
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            report = store.integrity_report()
            plan = RetentionPlanner(store, config.privacy).plan()
            report["retention_dry_run"] = {
                "media_cutoff": plan.media_cutoff.isoformat(),
                "evidence_cutoff": plan.evidence_cutoff.isoformat(),
                "batch_size": plan.batch_size,
            }
            passed = bool(
                report["sqlite_integrity"] == "ok"
                and not report["foreign_key_violations"]
                and not report["media"]["missing_evidence_ids"]
                and not report["media"]["checksum_mismatch_evidence_ids"]
            )
            report["status"] = "pass" if passed else "fail"
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if passed else 1
        finally:
            store.close()
    await CompanionRuntime(config).run()
    return 0


def main() -> None:
    args = _parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        logging.getLogger(__name__).exception("Egg companion stopped: %s", error)
        raise SystemExit(1) from error

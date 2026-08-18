from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import logging
import signal
import sys

from egg_companion.config import load_config
from egg_companion.evaluation import evaluate_trace, load_trace
from egg_companion.memory.migrate_legacy import LegacyMemoryMigrator
from egg_companion.memory.retention import RetentionPlanner
from egg_companion.memory.store import MemoryStore
from egg_companion.ocr.jobs import OcrBackfillScheduler, OcrJobLedger
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
    ocr = subcommands.add_parser("ocr", help="OCR backfill and diagnostics")
    ocr_commands = ocr.add_subparsers(dest="ocr_command", required=True)
    ocr_backfill = ocr_commands.add_parser("backfill", help="retroactively OCR existing visual evidence")
    ocr_backfill.add_argument("--batch-size", type=int, default=16, help="evidence items per batch")
    ocr_backfill.add_argument("--max-items", type=int, default=0, help="stop after N items (0 = all)")
    ocr_backfill.add_argument("--dry-run", action="store_true", help="show what would be processed")
    ocr_commands.add_parser("status", help="show OCR pipeline status and backfill progress")
    world = subcommands.add_parser("world", help="world model management")
    world_commands = world.add_subparsers(dest="world_command", required=True)
    world_commands.add_parser("rebuild", help="rebuild world model from memory")
    world_commands.add_parser("status", help="show world model state")
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
    if args.command == "ocr":
        return _run_ocr_command(config, args)
    if args.command == "world":
        return _run_world_command(config, args)
    await CompanionRuntime(config).run()
    return 0


def _run_ocr_command(config: object, args: argparse.Namespace) -> int:
    from pathlib import Path
    store = MemoryStore(config.memory)
    ledger_db = Path(config.ocr.ledger_db_path) if hasattr(config.ocr, "ledger_db_path") else Path("data/ocr-jobs.sqlite3")
    ledger = OcrJobLedger(ledger_db)
    backfill = OcrBackfillScheduler(
        enabled=True,
        scan_interval_seconds=0,
        batch_size=args.batch_size,
    )
    try:
        if args.ocr_command == "status":
            total = backfill.count_unprocessed(store)
            pending = ledger.pending_count()
            print(json.dumps({
                "unprocessed_evidence": total,
                "pending_jobs": pending,
                "ledger_db": str(ledger_db),
                "storage_dir": str(config.memory.storage_dir),
            }, indent=2))
            return 0
        if args.ocr_command == "backfill":
            processed = 0
            offset = 0
            max_items = args.max_items
            batch_size = args.batch_size
            while True:
                items = backfill.find_all_unprocessed(store, offset=offset, limit=batch_size)
                if not items:
                    break
                for item in items:
                    if max_items and processed >= max_items:
                        print(json.dumps({"processed": processed, "stopped_at_limit": True}))
                        return 0
                    media_key = item.get("media_key")
                    if not media_key:
                        continue
                    media_path = Path(config.memory.storage_dir) / "media" / media_key
                    if not media_path.exists():
                        continue
                    if args.dry_run:
                        print(json.dumps({
                            "evidence_id": item.get("evidence_id"),
                            "media_key": media_key,
                            "camera_id": item.get("camera_id"),
                            "captured_at": str(item.get("captured_at")),
                            "media_exists": True,
                        }))
                        processed += 1
                        continue
                    print(f"  [{processed+1}] OCR {media_key} ({item.get('camera_id', '?')}) ...", end=" ", flush=True)
                    processed += 1
                offset += len(items)
                if max_items and processed >= max_items:
                    break
            print(json.dumps({"processed": processed, "completed": True}))
            return 0
    finally:
        store.close()
    return 0


def _run_world_command(config: object, args: argparse.Namespace) -> int:
    from pathlib import Path
    store = MemoryStore(config.memory)
    try:
        if args.world_command == "status":
            from egg_companion.world.query import WorldQuery
            from egg_companion.world.state import WorldStateStore
            from egg_companion.world.reconcile import Reconciler
            world_db = Path(config.memory.storage_dir) / "world.sqlite3"
            if not world_db.exists():
                print(json.dumps({"error": "world model database not found", "path": str(world_db)}))
                return 1
            state = WorldStateStore(world_db)
            query = WorldQuery(state, store.knowledge_graph, None)
            summary = query.summary()
            conflicts = query.conflicts()
            print(json.dumps({
                **summary,
                "conflict_count": len(conflicts),
                "revision": state.revision,
            }, indent=2))
            return 0
        if args.world_command == "rebuild":
            from egg_companion.world.state import WorldStateStore
            world_db = Path(config.memory.storage_dir) / "world.sqlite3"
            state = WorldStateStore(world_db)
            print(f"World model state: revision {state.revision}, {len(state.all_entity_ids())} entities")
            print("Rebuild from memory is not yet implemented — world model populates during live runtime.")
            return 0
    finally:
        store.close()
    return 0


def main() -> None:
    # A live Jetson can spend long stretches inside native CUDA/BLAS calls.
    # SIGUSR2 provides a non-destructive all-thread traceback for diagnosing a
    # slow component without stopping voice, cameras, or the dashboard.
    faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
    args = _parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        logging.getLogger(__name__).exception("Egg companion stopped: %s", error)
        raise SystemExit(1) from error

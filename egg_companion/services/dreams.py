from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from egg_companion.config import DreamsConfig
from egg_companion.services.identity import IdentityLibrary


class AdaFaceEmbedder:
    """Pinned, offline CVLFace AdaFace IR18 inference on aligned face crops."""

    def __init__(self, config: DreamsConfig) -> None:
        self.config = config
        self.model_path = self._resolve_model_path(config.model_path)
        self._model = None
        self._torch = None
        self.device = "unloaded"

    @staticmethod
    def _resolve_model_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        working = (Path.cwd() / path).resolve()
        if working.exists():
            return working
        return (Path(__file__).resolve().parents[2] / path).resolve()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from safetensors.torch import load_file

        source = self.model_path / "models" / "__init__.py"
        weights = self.model_path / "model.safetensors"
        if not source.is_file() or not weights.is_file():
            raise RuntimeError(
                f"offline AdaFace package is incomplete at {self.model_path}; "
                "model.safetensors and models/ are required"
            )
        package_name = "_egg_cvlface_models"
        module = sys.modules.get(package_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                package_name,
                source,
                submodule_search_locations=[str(source.parent)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("unable to load the local CVLFace model package")
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
        model_config = SimpleNamespace(
            color_space="RGB",
            freeze=True,
            input_size=[3, 112, 112],
            name="ir18",
            output_dim=512,
            start_from="",
            yaml_path="models/iresnet/configs/v1_ir18.yaml",
        )
        model = module.get_model(model_config)
        state = {
            key.removeprefix("model."): value
            for key, value in load_file(str(weights), device="cpu").items()
        }
        model.load_state_dict(state, strict=True)
        requested = self.config.device
        device = requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        model = model.eval().to(device)
        if self.config.use_half_precision and device.startswith("cuda"):
            model = model.half()
        self._model = model
        self._torch = torch
        self.device = device

    def embed(self, jpeg_images: list[bytes]) -> np.ndarray:
        # PyTorch's aarch64 CPU model loader can retain the GIL for long
        # stretches even when called through asyncio.to_thread. Keep the
        # dashboard, live ingress, and camera tasks isolated from dream work.
        if self.config.device == "cpu":
            return self._embed_in_worker(jpeg_images)
        return self._embed_local(jpeg_images)

    def _embed_in_worker(self, jpeg_images: list[bytes]) -> np.ndarray:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_adaface_embed_worker,
            args=(self.config.model_dump(mode="json"), jpeg_images, child),
            name="egg-adaface-dream",
            daemon=True,
        )
        process.start()
        child.close()
        deadline = time.monotonic() + 300
        try:
            while not parent.poll(0.25):
                if not process.is_alive():
                    process.join(timeout=1)
                    raise RuntimeError(
                        f"AdaFace worker exited without a result ({process.exitcode})"
                    )
                if time.monotonic() >= deadline:
                    process.terminate()
                    process.join(timeout=5)
                    raise TimeoutError("AdaFace worker exceeded its 300 second bound")
            status, payload, device = parent.recv()
            process.join(timeout=5)
            if status != "ok":
                raise RuntimeError(str(payload))
            self.device = str(device)
            return np.asarray(payload, dtype=np.float32)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def _embed_local(self, jpeg_images: list[bytes]) -> np.ndarray:
        import cv2

        self._load()
        assert self._model is not None and self._torch is not None
        torch = self._torch
        inputs: list[np.ndarray] = []
        for payload in jpeg_images:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("retained face evidence is not a decodable JPEG")
            image = cv2.cvtColor(
                cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            inputs.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.config.batch_size):
                tensor = torch.from_numpy(np.stack(inputs[start : start + self.config.batch_size]))
                tensor = tensor.to(self.device, non_blocking=True)
                dtype = next(self._model.parameters()).dtype
                tensor = tensor.to(dtype=dtype).div_(127.5).sub_(1.0)
                embedding = torch.nn.functional.normalize(self._model(tensor).float(), dim=1)
                outputs.append(embedding.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, 512), dtype=np.float32)


def _adaface_embed_worker(
    config_values: dict[str, object], jpeg_images: list[bytes], connection: object
) -> None:
    """Run heavyweight CPU inference outside the latency-sensitive daemon."""

    try:
        embedder = AdaFaceEmbedder(DreamsConfig.model_validate(config_values))
        embeddings = embedder._embed_local(jpeg_images)
        connection.send(("ok", embeddings, embedder.device))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}", "failed"))
    finally:
        connection.close()


class MobileFaceNetEmbedder:
    """Independent InsightFace comparison model using bounded CPU ONNX inference."""

    def __init__(self, model_path: str | None) -> None:
        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self._session = None
        self.device = "unloaded"

    @property
    def available(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self.available:
            raise RuntimeError("the optional InsightFace comparison model is unavailable")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.device = "cpu"

    def embed(self, jpeg_images: list[bytes]) -> np.ndarray:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_mobileface_embed_worker,
            args=(str(self.model_path or ""), jpeg_images, child),
            name="egg-mobileface-dream",
            daemon=True,
        )
        process.start()
        child.close()
        deadline = time.monotonic() + 120
        try:
            while not parent.poll(0.25):
                if not process.is_alive():
                    process.join(timeout=1)
                    raise RuntimeError(
                        f"MobileFaceNet worker exited without a result ({process.exitcode})"
                    )
                if time.monotonic() >= deadline:
                    process.terminate()
                    process.join(timeout=5)
                    raise TimeoutError("MobileFaceNet worker exceeded its 120 second bound")
            status, payload, device = parent.recv()
            process.join(timeout=5)
            if status != "ok":
                raise RuntimeError(str(payload))
            self.device = str(device)
            return np.asarray(payload, dtype=np.float32)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def _embed_local(self, jpeg_images: list[bytes]) -> np.ndarray:
        import cv2

        self._load()
        assert self._session is not None
        inputs: list[np.ndarray] = []
        for payload in jpeg_images:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("retained face evidence is not a decodable JPEG")
            image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
            image = (image.astype(np.float32) - 127.5) / 127.5
            inputs.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if not inputs:
            return np.empty((0, 512), dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        embedding = self._session.run(
            None, {input_name: np.stack(inputs, dtype=np.float32)}
        )[0].astype(np.float32)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norms, 1e-12)


def _mobileface_embed_worker(
    model_path: str, jpeg_images: list[bytes], connection: object
) -> None:
    """Keep ONNX CPU thread pools out of the real-time companion process."""

    try:
        embedder = MobileFaceNetEmbedder(model_path)
        embeddings = embedder._embed_local(jpeg_images)
        connection.send(("ok", embeddings, embedder.device))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}", "failed"))
    finally:
        connection.close()


class IdentityDreamEngine:
    """Quality-aware, constraint-aware identity consolidation during idle periods."""

    usage_notice = (
        "AdaFace IR18 / WebFace4M research weights; follow the model card and training-data "
        "license for the intended deployment. Runtime loading is local and offline."
    )

    def __init__(self, config: DreamsConfig, identities: IdentityLibrary) -> None:
        self.config = config
        self.identities = identities
        directory = Path(identities.config.storage_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(
            directory / "identity-dreams.sqlite3", check_same_thread=False
        )
        self._database.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._embedder = AdaFaceEmbedder(config)
        self._comparison_embedder = MobileFaceNetEmbedder(config.comparison_model_path)
        self._state = "idle" if config.enabled else "disabled"
        self._active_run_id: str | None = None
        self._next_scheduled_at: str | None = None
        self._create_schema()
        self._recover_interrupted_runs()

    def _create_schema(self) -> None:
        self._database.executescript(
            """
            CREATE TABLE IF NOT EXISTS dream_runs (
                run_id TEXT PRIMARY KEY,
                requested_by TEXT NOT NULL,
                state TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                device TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                profiles_examined INTEGER NOT NULL DEFAULT 0,
                samples_embedded INTEGER NOT NULL DEFAULT 0,
                proposals INTEGER NOT NULL DEFAULT 0,
                merges INTEGER NOT NULL DEFAULT 0,
                conflicts_blocked INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL,
                error TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS dream_runs_started ON dream_runs(started_at DESC);
            CREATE TABLE IF NOT EXISTS dream_candidates (
                run_id TEXT NOT NULL REFERENCES dream_runs(run_id) ON DELETE CASCADE,
                left_id TEXT NOT NULL,
                right_id TEXT NOT NULL,
                modern_similarity REAL NOT NULL,
                legacy_similarity REAL NOT NULL,
                left_margin REAL NOT NULL,
                right_margin REAL NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                canonical_id TEXT,
                alias_id TEXT,
                PRIMARY KEY(run_id, left_id, right_id)
            );
            CREATE INDEX IF NOT EXISTS dream_candidates_run ON dream_candidates(run_id, decision);
            """
        )
        columns = {
            str(row[1]) for row in self._database.execute("PRAGMA table_info(dream_candidates)")
        }
        if "comparison_similarity" not in columns:
            self._database.execute(
                "ALTER TABLE dream_candidates ADD COLUMN comparison_similarity REAL"
            )
        self._database.commit()

    def _recover_interrupted_runs(self) -> None:
        """Close run rows left behind when a process was stopped mid-inference."""

        completed = datetime.now(timezone.utc)
        rows = self._database.execute(
            "SELECT run_id, started_at FROM dream_runs WHERE state='running'"
        ).fetchall()
        for row in rows:
            try:
                started = datetime.fromisoformat(str(row["started_at"]))
                duration = max(0.0, (completed - started).total_seconds())
            except ValueError:
                duration = 0.0
            self._database.execute(
                """UPDATE dream_runs SET state='interrupted', completed_at=?,
                duration_seconds=?, error=? WHERE run_id=?""",
                (
                    completed.isoformat(),
                    round(duration, 3),
                    "process stopped before the dream pass completed",
                    str(row["run_id"]),
                ),
            )
        if rows:
            self._database.commit()

    def set_next_scheduled_at(self, value: datetime | None) -> None:
        with self._lock:
            self._next_scheduled_at = value.isoformat() if value is not None else None

    def annotate_run(self, run_id: str, supplemental: dict[str, object]) -> None:
        """Attach post-identity replay outcomes to the durable dream audit row."""
        with self._lock:
            row = self._database.execute(
                "SELECT details_json FROM dream_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except json.JSONDecodeError:
                details = {}
            details.update(supplemental)
            self._database.execute(
                "UPDATE dream_runs SET details_json=? WHERE run_id=?",
                (json.dumps(details, sort_keys=True), run_id),
            )
            self._database.commit()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            runs = [dict(row) for row in self._database.execute(
                "SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT 12"
            ).fetchall()]
            candidates = [dict(row) for row in self._database.execute(
                """SELECT candidate.* FROM dream_candidates candidate
                JOIN dream_runs run ON run.run_id=candidate.run_id
                ORDER BY run.started_at DESC,
                CASE candidate.decision WHEN 'merged' THEN 0 WHEN 'blocked' THEN 1 ELSE 2 END,
                candidate.modern_similarity DESC LIMIT 200"""
            ).fetchall()]
            for run in runs:
                run["details"] = json.loads(str(run.pop("details_json") or "{}"))
            return {
                "enabled": self.config.enabled,
                "state": self._state,
                "active_run_id": self._active_run_id,
                "next_scheduled_at": self._next_scheduled_at,
                "model": {
                    "id": self.config.model_id,
                    "revision": self.config.model_revision,
                    "architecture": "AdaFace IR18",
                    "device": self._embedder.device,
                    "configured_device": self.config.device,
                    "offline": True,
                    "path": str(self._embedder.model_path),
                    "ready": (
                        (self._embedder.model_path / "model.safetensors").is_file()
                        and (self._embedder.model_path / "models" / "__init__.py").is_file()
                    ),
                    "usage_notice": self.usage_notice,
                    "comparison": {
                        "id": self.config.comparison_model_id,
                        "path": str(self._comparison_embedder.model_path or ""),
                        "ready": self._comparison_embedder.available,
                        "device": self._comparison_embedder.device,
                    },
                },
                "policy": {
                    "idle_seconds": self.config.idle_seconds,
                    "interval_min_seconds": self.config.interval_min_seconds,
                    "interval_max_seconds": self.config.interval_max_seconds,
                    "convergence_interval_seconds": self.config.convergence_interval_seconds,
                    "proposal_similarity": self.config.proposal_similarity,
                    "modern_merge_similarity": self.config.modern_merge_similarity,
                    "modern_strong_similarity": self.config.modern_strong_similarity,
                    "legacy_merge_similarity": self.config.legacy_merge_similarity,
                    "legacy_strong_similarity": self.config.legacy_strong_similarity,
                    "legacy_similarity_floor": self.config.legacy_similarity_floor,
                    "comparison_merge_similarity": self.config.comparison_merge_similarity,
                    "comparison_strong_similarity": self.config.comparison_strong_similarity,
                    "comparison_similarity_floor": self.config.comparison_similarity_floor,
                    "minimum_model_votes": self.config.minimum_model_votes,
                    "separated_modern_similarity": self.config.separated_modern_similarity,
                    "separated_legacy_floor": self.config.separated_legacy_floor,
                    "mutual_neighbor_margin": self.config.mutual_neighbor_margin,
                    "reciprocal_neighbor_rank": self.config.reciprocal_neighbor_rank,
                    "coobservation_min_confirmations": self.config.coobservation_min_confirmations,
                    "auto_merge_enabled": self.config.auto_merge_enabled,
                    "constraints": [
                        "two-of-three AdaFace, SFace, and MobileFaceNet consensus",
                        "dense-cluster edges do not require fragile mutual-nearest ranking",
                        "repeated or spatially distinct co-observation veto",
                        "compatible user-provided names",
                        "reversible aliases; source evidence retained",
                    ],
                },
                "runs": runs,
                "candidates": candidates,
            }

    def run(
        self,
        conflicting_pairs: set[tuple[str, str]] | None = None,
        requested_by: str = "scheduler",
        verifier: Callable[[str, str, bool], bool | None] | None = None,
    ) -> dict[str, object]:
        """verifier, if given, is called as verifier(alias_id, canonical_id,
        mandatory) for every candidate the embedding consensus alone
        decided to merge, before create_alias is actually invoked --
        typically a synchronous bridge into an Ornith VLM visual
        confirmation (see runtime.CompanionRuntime.run_identity_dream).
        It returns True (confirmed, proceed), False (VLM disagrees, block),
        or None (verification unavailable/errored). `mandatory` is True
        for the highest-stakes case -- exactly one profile in the pair is
        named, so an unconfirmed merge risks silently mislabeling a real
        person -- in which case None is treated as a block (fail closed)
        rather than the normal fail-open behavior for two anonymous
        fragments, where a missed cycle just tries again later.
        """
        if not self.config.enabled:
            raise RuntimeError("identity dreaming is disabled")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("an identity dream is already running")
        run_id = f"dream-{uuid4()}"
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        with self._lock:
            self._state = "dreaming"
            self._active_run_id = run_id
            self._database.execute(
                """INSERT INTO dream_runs
                (run_id, requested_by, state, model_id, model_revision, started_at)
                VALUES (?, ?, 'running', ?, ?, ?)""",
                (
                    run_id,
                    requested_by,
                    self.config.model_id,
                    self.config.model_revision,
                    started.isoformat(),
                ),
            )
            self._database.commit()
        try:
            result = self._run_once(run_id, conflicting_pairs or set(), verifier)
            result["duration_seconds"] = round(time.monotonic() - started_clock, 3)
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._finish_run(run_id, "completed", result)
            return result
        except Exception as error:
            result = {
                "run_id": run_id,
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": round(time.monotonic() - started_clock, 3),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._finish_run(run_id, "failed", result)
            raise
        finally:
            with self._lock:
                self._state = "idle"
                self._active_run_id = None
            self._run_lock.release()

    def _run_once(
        self,
        run_id: str,
        conflicting_pairs: set[tuple[str, str]],
        verifier: Callable[[str, str, bool], bool | None] | None = None,
    ) -> dict[str, object]:
        profiles = self.identities.migration_profiles()
        alias_rows = self.identities.alias_mappings()
        alias_map = {
            str(row["alias_id"]): str(row["canonical_id"]) for row in alias_rows
        }

        def canonical(profile_id: str) -> str:
            seen: set[str] = set()
            while profile_id in alias_map and profile_id not in seen:
                seen.add(profile_id)
                profile_id = alias_map[profile_id]
            return profile_id

        profile_by_id = {str(profile["profile_id"]): profile for profile in profiles}
        groups: dict[str, list[str]] = defaultdict(list)
        for profile in profiles:
            if (
                profile.get("face_embedding") is not None
                and profile.get("kind") != "invalid-face"
                and int(profile.get("valid_face_samples", 1) or 0) > 0
            ):
                groups[canonical(str(profile["profile_id"]))].append(str(profile["profile_id"]))
        samples_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
        valid_profile_ids = {member for members in groups.values() for member in members}
        for sample in self.identities.face_sample_snapshot():
            if str(sample["profile_id"]) not in valid_profile_ids:
                continue
            group_id = canonical(str(sample["profile_id"]))
            if group_id in groups:
                samples_by_group[group_id].append(sample)
        ids = sorted(group_id for group_id in groups if samples_by_group[group_id])
        flat_samples = [sample for group_id in ids for sample in samples_by_group[group_id]]
        self._update_run_progress(run_id, len(ids), len(flat_samples))
        embeddings = self._embedder.embed(
            [bytes(sample["image_jpeg"]) for sample in flat_samples]
        )
        comparison_embeddings = (
            self._comparison_embedder.embed(
                [bytes(sample["image_jpeg"]) for sample in flat_samples]
            )
            if self._comparison_embedder.available
            else None
        )
        modern_templates: list[np.ndarray] = []
        legacy_templates: list[np.ndarray] = []
        comparison_templates: list[np.ndarray] = []
        cursor = 0
        for group_id in ids:
            samples = samples_by_group[group_id]
            count = len(samples)
            quality = np.array(
                [max(0.25, min(1.0, float(sample["quality"]))) ** 2 for sample in samples],
                dtype=np.float32,
            )
            modern_templates.append(self._normalized((embeddings[cursor : cursor + count] * quality[:, None]).sum(axis=0)))
            legacy = np.stack([np.asarray(sample["sface_embedding"], dtype=np.float32) for sample in samples])
            legacy_templates.append(self._normalized((legacy * quality[:, None]).sum(axis=0)))
            if comparison_embeddings is not None:
                comparison_templates.append(
                    self._normalized(
                        (comparison_embeddings[cursor : cursor + count] * quality[:, None]).sum(axis=0)
                    )
                )
            cursor += count
        if len(ids) < 2:
            return {
                "run_id": run_id,
                "profiles_examined": len(ids),
                "samples_embedded": len(flat_samples),
                "proposals": 0,
                "merges": 0,
                "conflicts_blocked": 0,
                "aliases": [],
            }
        modern_matrix = self._cosine_matrix(modern_templates)
        legacy_matrix = self._cosine_matrix(legacy_templates)
        comparison_matrix = (
            self._cosine_matrix(comparison_templates)
            if comparison_templates
            else None
        )
        np.fill_diagonal(modern_matrix, -2.0)
        np.fill_diagonal(legacy_matrix, -2.0)
        if comparison_matrix is not None:
            np.fill_diagonal(comparison_matrix, -2.0)
        order = np.argsort(modern_matrix, axis=1)[:, ::-1]
        names: dict[str, set[str]] = {}
        for group_id in ids:
            names[group_id] = {
                str(profile_by_id[member]["name"]).strip().casefold()
                for member in groups[group_id]
                if profile_by_id.get(member, {}).get("name")
            }
        conflicts = {
            tuple(sorted((canonical(str(left)), canonical(str(right)))))
            for left, right in conflicting_pairs
            if canonical(str(left)) != canonical(str(right))
        }
        candidates: list[dict[str, object]] = []
        mergeable: list[tuple[float, int, int, dict[str, object]]] = []
        conflicts_blocked = 0
        reciprocal_rank = min(self.config.reciprocal_neighbor_rank, len(ids) - 1)
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                modern = float(modern_matrix[left, right])
                if modern < self.config.proposal_similarity:
                    continue
                legacy = float(legacy_matrix[left, right])
                comparison = (
                    float(comparison_matrix[left, right])
                    if comparison_matrix is not None
                    else None
                )
                left_competitors = [
                    float(modern_matrix[left, index])
                    for index in order[left]
                    if int(index) not in {left, right}
                ]
                right_competitors = [
                    float(modern_matrix[right, index])
                    for index in order[right]
                    if int(index) not in {left, right}
                ]
                left_margin = modern - max(left_competitors, default=-1.0)
                right_margin = modern - max(right_competitors, default=-1.0)
                left_rank = int(np.flatnonzero(order[left] == right)[0])
                right_rank = int(np.flatnonzero(order[right] == left)[0])
                reciprocal = left_rank < reciprocal_rank and right_rank < reciprocal_rank
                pair = tuple(sorted((ids[left], ids[right])))
                name_conflict = bool(names[ids[left]] and names[ids[right]] and names[ids[left]] != names[ids[right]])
                # Exactly one side already has a user-visible name: absorbing
                # it into an unnamed fragment (or vice versa) is the
                # highest-stakes failure mode -- it silently mislabels a
                # real, already-identified person. A same-name match on
                # both sides is reinforcing evidence, not risk, so this is
                # deliberately narrower than name_conflict above.
                asymmetric_naming = bool(names[ids[left]]) != bool(names[ids[right]])
                model_votes = int(modern >= self.config.modern_merge_similarity) + int(
                    legacy >= self.config.legacy_merge_similarity
                )
                required_votes = 2
                if comparison is not None:
                    model_votes += int(
                        comparison >= self.config.comparison_merge_similarity
                    )
                    required_votes = self.config.minimum_model_votes
                agrees = model_votes >= required_votes or (
                    modern >= self.config.modern_strong_similarity
                    and (
                        legacy >= self.config.legacy_similarity_floor
                        or comparison is not None
                        and comparison >= self.config.comparison_similarity_floor
                    )
                ) or (
                    legacy >= self.config.legacy_strong_similarity
                    and modern >= self.config.proposal_similarity
                ) or (
                    comparison is not None
                    and comparison >= self.config.comparison_strong_similarity
                    and modern >= self.config.proposal_similarity
                ) or (
                    modern >= self.config.separated_modern_similarity
                    and legacy >= self.config.separated_legacy_floor
                    and min(left_margin, right_margin)
                    >= self.config.mutual_neighbor_margin
                )
                if pair in conflicts:
                    decision, reason = "blocked", "co_observation_conflict"
                    conflicts_blocked += 1
                elif name_conflict:
                    decision, reason = "blocked", "incompatible_user_names"
                elif not agrees:
                    decision, reason = "review", "embedding_models_do_not_agree"
                elif not self.config.auto_merge_enabled:
                    decision, reason = "review", "automatic_merge_disabled"
                else:
                    decision, reason = "merge", "quality_aggregated_multimodel_consensus"
                if decision == "merge" and verifier is not None:
                    # Embedding consensus alone previously produced badly
                    # miscalibrated merges (documented: an unnamed fragment
                    # absorbing a well-established named profile at 0.41
                    # similarity) -- a VLM visual pass is the actual
                    # reasoning step that should gate an irreversible-ish
                    # canonical/alias decision, not just cosine similarity.
                    confirmed = verifier(ids[left], ids[right], asymmetric_naming)
                    if confirmed is False or (confirmed is None and asymmetric_naming):
                        decision = "review"
                        reason = (
                            "vlm_rejected_merge" if confirmed is False
                            else "vlm_unavailable_for_named_profile"
                        )
                candidate = {
                    "run_id": run_id,
                    "left_id": ids[left],
                    "right_id": ids[right],
                    "modern_similarity": round(modern, 6),
                    "legacy_similarity": round(legacy, 6),
                    "comparison_similarity": round(comparison, 6) if comparison is not None else None,
                    "left_margin": round(left_margin, 6),
                    "right_margin": round(right_margin, 6),
                    "decision": decision,
                    "reason": reason,
                    "canonical_id": None,
                    "alias_id": None,
                }
                candidates.append(candidate)
                if decision == "merge":
                    mergeable.append((modern, left, right, candidate))
        new_aliases: list[dict[str, object]] = []
        for modern, left, right, candidate in sorted(mergeable, reverse=True):
            left_id, right_id = ids[left], ids[right]
            canonical_id, alias_id = self._canonical_choice(
                left_id, right_id, groups, profile_by_id
            )
            mapping = self.identities.create_alias(
                alias_id,
                canonical_id,
                modern,
                "dream_multimodel_template_consensus",
                conflicting_pairs,
            )
            if mapping is None:
                left_source = self.identities.identity_timeline_source(left_id)
                right_source = self.identities.identity_timeline_source(right_id)
                if left_source and right_source and left_source["id"] == right_source["id"]:
                    candidate["decision"] = "consolidated"
                    candidate["reason"] = "joined_by_stronger_cluster_edge"
                    candidate["canonical_id"] = str(left_source["id"])
                else:
                    candidate["decision"] = "blocked"
                    candidate["reason"] = "identity_changed_during_dream"
                continue
            candidate["decision"] = "merged"
            candidate["canonical_id"] = mapping["canonical_id"]
            candidate["alias_id"] = mapping["alias_id"]
            new_aliases.append(mapping)
        final_alias_map = {
            str(row["alias_id"]): str(row["canonical_id"])
            for row in self.identities.alias_mappings()
        }

        def final_canonical(profile_id: str) -> str:
            seen: set[str] = set()
            while profile_id in final_alias_map and profile_id not in seen:
                seen.add(profile_id)
                profile_id = final_alias_map[profile_id]
            return profile_id

        # Edges inside a cluster are no longer unresolved simply because a
        # stronger edge performed the actual alias operation first.
        for candidate in candidates:
            left_final = final_canonical(str(candidate["left_id"]))
            right_final = final_canonical(str(candidate["right_id"]))
            if left_final == right_final and candidate["decision"] != "merged":
                candidate["decision"] = "consolidated"
                candidate["reason"] = "joined_by_stronger_cluster_edge"
                candidate["canonical_id"] = left_final
        with self._lock:
            self._database.executemany(
                """INSERT INTO dream_candidates
                (run_id, left_id, right_id, modern_similarity, legacy_similarity, comparison_similarity,
                left_margin, right_margin, decision, reason, canonical_id, alias_id)
                VALUES (:run_id, :left_id, :right_id, :modern_similarity, :legacy_similarity, :comparison_similarity,
                :left_margin, :right_margin, :decision, :reason, :canonical_id, :alias_id)""",
                candidates,
            )
            self._database.commit()
        return {
            "run_id": run_id,
            "profiles_examined": len(ids),
            "samples_embedded": len(flat_samples),
            "proposals": len(candidates),
            "merges": len(new_aliases),
            "conflicts_blocked": conflicts_blocked,
            "aliases": new_aliases,
            "model_device": self._embedder.device,
        }

    def _update_run_progress(
        self, run_id: str, profiles_examined: int, samples_embedded: int
    ) -> None:
        with self._lock:
            self._database.execute(
                """UPDATE dream_runs SET device=?, profiles_examined=?, samples_embedded=?
                WHERE run_id=?""",
                (
                    self.config.device,
                    profiles_examined,
                    samples_embedded,
                    run_id,
                ),
            )
            self._database.commit()

    def _finish_run(self, run_id: str, state: str, result: dict[str, object]) -> None:
        details = dict(result)
        aliases = details.pop("aliases", [])
        with self._lock:
            self._database.execute(
                """UPDATE dream_runs SET state=?, device=?, completed_at=?,
                profiles_examined=?, samples_embedded=?, proposals=?, merges=?,
                conflicts_blocked=?, duration_seconds=?, error=?, details_json=? WHERE run_id=?""",
                (
                    state,
                    self._embedder.device,
                    result.get("completed_at"),
                    int(result.get("profiles_examined") or 0),
                    int(result.get("samples_embedded") or 0),
                    int(result.get("proposals") or 0),
                    int(result.get("merges") or 0),
                    int(result.get("conflicts_blocked") or 0),
                    float(result.get("duration_seconds") or 0),
                    result.get("error"),
                    json.dumps({**details, "aliases": aliases}, sort_keys=True),
                    run_id,
                ),
            )
            self._database.commit()

    @staticmethod
    def _canonical_choice(
        left_id: str,
        right_id: str,
        groups: dict[str, list[str]],
        profiles: dict[str, dict[str, object]],
    ) -> tuple[str, str]:
        def rank(group_id: str) -> tuple[int, int, int, str]:
            members = [profiles[member] for member in groups[group_id]]
            return (
                int(any(member.get("name") for member in members)),
                sum(int(member.get("samples") or 0) * max(1, int(member.get("sightings") or 0)) for member in members),
                sum(int(member.get("samples") or 0) for member in members),
                "".join(chr(0x10FFFF - ord(char)) for char in group_id),
            )

        canonical_id = max((left_id, right_id), key=rank)
        return canonical_id, right_id if canonical_id == left_id else left_id

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("dream identity template must not be empty")
        return np.asarray(vector, dtype=np.float32) / norm

    @staticmethod
    def _cosine_matrix(templates: list[np.ndarray]) -> np.ndarray:
        """Compute the small gallery matrix without a global BLAS thread pool.

        OpenBLAS matrix multiplication can deadlock on Jetson after native
        PyTorch/ONNX worker processes have exited. The gallery is small enough
        that NumPy's direct contraction is both bounded and faster in practice.
        """

        values = np.stack(templates).astype(np.float32, copy=False)
        return np.einsum("ik,jk->ij", values, values, optimize=False)

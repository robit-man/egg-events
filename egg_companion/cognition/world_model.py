from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from egg_companion.config import DefaultModeConfig
from egg_companion.memory.store import MemoryStore


_TERM_STOPWORDS = {
    "about", "after", "again", "also", "because", "could", "from", "have",
    "into", "just", "like", "some", "that", "their", "there", "these",
    "they", "this", "what", "when", "where", "which", "with", "would",
    "your", "youre", "will", "then", "than", "them", "were", "been",
}


class WorldModelSynthesizer:
    """Project evidence-grounded graph motifs into a revisable meta-graph.

    The output is an inspectable working model, not hidden chain-of-thought.
    Every abstraction records its supporting entity and episode IDs, and
    repeated co-occurrence is always labelled non-causal.
    """

    DOCUMENT_TITLES = {
        "world-model": "World model",
        "my-story": "My story",
        "communication-strategy": "Communication strategy",
        "reflective-working-set": "Reflective working set",
    }
    NARRATIVE_SCHEMA_REVISION = 4

    def __init__(self, store: MemoryStore, config: DefaultModeConfig) -> None:
        self.store = store
        self.config = config

    def update(
        self,
        replayed_entity_ids: list[str],
        reflection_ids: list[str],
        at: datetime,
    ) -> dict[str, object]:
        inventory = self.store.cognitive_inventory(
            max(self.config.entity_summary_limit, self.config.replay_limit)
        )[: self.config.entity_summary_limit]
        pairs = self.store.recurrent_entity_pairs(
            self.config.meta_graph_min_confirmations,
            self.config.meta_graph_limit * 4,
            self.config.meta_graph_period_seconds,
        )
        pairs = self._deduplicate_semantic_pairs(pairs)[
            : self.config.meta_graph_limit
        ]
        abstractions = [self._project_association(pair, at) for pair in pairs]
        retired = self.store.retire_inactive_meta_graph(
            [str(item["abstraction_id"]) for item in abstractions], at
        )
        for item in inventory:
            self.store.update_entity_summary(
                str(item["entity_id"]), self._entity_summary(item), at
            )

        outcomes = self.store.recent_interaction_outcomes(200)
        history = self.store.conversation_history(500)
        conflicts = self.store.conflicting_claims(20)
        episodes = self.store.recent_episodes(12)
        daily_narratives = self.store.recent_daily_narratives(7)
        documents = self._document_contents(
            inventory,
            abstractions,
            outcomes,
            history,
            conflicts,
            episodes,
            daily_narratives,
        )
        source_ids = list(
            dict.fromkeys(
                [*replayed_entity_ids, *reflection_ids]
                + [str(item["abstraction_id"]) for item in abstractions]
                + [str(item["entity_id"]) for item in daily_narratives]
            )
        )
        self.store.upsert_entity(
            "agent",
            "Egg",
            {
                "role": "local embodied companion",
                "epistemic_status": "system identity",
            },
            "agent:egg",
            now=at,
        )
        document_records: list[dict[str, object]] = []
        for kind, content in documents.items():
            record = self.store.upsert_cognitive_document(
                kind,
                self.DOCUMENT_TITLES[kind],
                content,
                self._document_confidence(inventory, abstractions),
                source_ids,
                at,
            )
            document_records.append(record)
            relation = (
                "guides_communication"
                if kind == "communication-strategy"
                else "maintains"
            )
            self._edge(
                "agent:egg",
                relation,
                str(record["document_id"]),
                1.0,
                1,
                at,
                {"derived": True, "document_kind": kind},
            )

        world_document_id = "cognitive-document:world-model"
        working_document_id = "cognitive-document:reflective-working-set"
        for abstraction in abstractions:
            self._edge(
                str(abstraction["abstraction_id"]),
                "informs_world_model",
                world_document_id,
                float(abstraction["confidence"]),
                int(abstraction["confirmations"]),
                at,
                {"derived": True, "epistemic_status": "noncausal_association"},
            )
        for reflection_id in reflection_ids:
            self._edge(
                reflection_id,
                "informs_working_set",
                working_document_id,
                0.7,
                1,
                at,
                {"derived": True, "source": "default-mode-replay"},
            )
        self._edge(
            world_document_id,
            "grounds_narrative",
            "cognitive-document:my-story",
            0.9,
            1,
            at,
            {"derived": True},
        )
        self._edge(
            working_document_id,
            "updates_strategy",
            "cognitive-document:communication-strategy",
            0.75,
            1,
            at,
            {"derived": True},
        )
        return {
            "abstractions_projected": len(abstractions),
            "abstractions_retired": retired,
            "abstraction_ids": [item["abstraction_id"] for item in abstractions],
            "entity_summaries_updated": len(inventory),
            "documents": document_records,
        }

    def replay_dream(
        self, dream_result: dict[str, object], at: datetime | None = None
    ) -> dict[str, object]:
        """Turn one identity dream into dated replay, story, and graph revisions."""
        replayed_at = at or datetime.now(timezone.utc)
        run_id = str(dream_result.get("run_id") or f"untracked-{int(replayed_at.timestamp())}")
        aliases = dream_result.get("aliases")
        alias_rows = aliases if isinstance(aliases, list) else []
        affected_entity_ids = list(
            dict.fromkeys(
                str(mapping.get(key))
                for mapping in alias_rows
                if isinstance(mapping, dict)
                for key in ("canonical_id", "alias_id")
                if mapping.get(key)
            )
        )
        job_id = self.store.create_job(
            "dream-chronological-replay",
            {
                "dream_run_id": run_id,
                "affected_entity_ids": affected_entity_ids,
            },
        )
        self.store.update_job(job_id, "running")
        try:
            local_timezone, timezone_name = self._narrative_timezone()
            affected_timestamps = (
                self.store.narrative_candidate_timestamps(affected_entity_ids)
                if affected_entity_ids
                else []
            )
            history_timestamps = self.store.narrative_history_boundaries()
            affected_days: set[date] = set()
            history_days: set[date] = set()
            for value in affected_timestamps:
                parsed = self._parse_datetime(value)
                if parsed is not None:
                    affected_days.add(parsed.astimezone(local_timezone).date())
            for value in history_timestamps:
                parsed = self._parse_datetime(value)
                if parsed is not None:
                    history_days.add(parsed.astimezone(local_timezone).date())

            existing_days = {
                parsed
                for item in self.store.recent_daily_narratives(3650)
                if isinstance(item.get("metadata"), dict)
                and int(item["metadata"].get("narrative_schema_revision") or 0)
                >= self.NARRATIVE_SCHEMA_REVISION
                for parsed in [
                    self._parse_local_date(item["metadata"].get("local_date"))
                ]
                if parsed is not None
            }
            unreviewed_days = history_days - existing_days
            latest_day = max(history_days) if history_days else None
            mandatory_days = set(affected_days)
            if latest_day is not None:
                mandatory_days.add(latest_day)
            maximum_days = max(1, int(self.config.narrative_replay_max_days))
            if len(mandatory_days) > maximum_days:
                retained_mandatory = sorted(mandatory_days)[: maximum_days - 1]
                if latest_day is not None:
                    retained_mandatory.append(latest_day)
                mandatory_days = set(retained_mandatory)
            backlog_slots = max(0, maximum_days - len(mandatory_days))
            oldest_unreviewed = sorted(unreviewed_days - mandatory_days)
            days = sorted(
                {
                    *mandatory_days,
                    *oldest_unreviewed[:backlog_slots],
                }
            )

            self.store.upsert_entity(
                "agent",
                "Egg",
                {
                    "role": "local embodied companion",
                    "epistemic_status": "system identity",
                },
                "agent:egg",
                now=replayed_at,
            )
            dream_node_id = f"dream-replay:{run_id}"
            self.store.upsert_entity(
                "dream_replay",
                f"Dream replay · {replayed_at.astimezone(local_timezone).strftime('%Y-%m-%d %H:%M')}",
                {
                    "dream_run_id": run_id,
                    "requested_by": dream_result.get("requested_by"),
                    "profiles_examined": int(dream_result.get("profiles_examined") or 0),
                    "identity_merges": int(dream_result.get("merges") or 0),
                    "affected_entity_ids": affected_entity_ids,
                    "replayed_at": replayed_at.isoformat(),
                    "epistemic_status": "offline_evidence_replay",
                },
                dream_node_id,
                now=replayed_at,
            )
            self._edge(
                "agent:egg",
                "enters_dream_replay",
                dream_node_id,
                1.0,
                1,
                replayed_at,
                {"derived": True, "dream_run_id": run_id},
            )

            records: list[dict[str, object]] = []
            all_replayed_entities: list[str] = []
            for local_day in days:
                start_local = datetime.combine(local_day, time.min, local_timezone)
                end_local = start_local + timedelta(days=1)
                events = self.store.chronological_evidence(
                    start_local.astimezone(timezone.utc),
                    end_local.astimezone(timezone.utc),
                    limit=50000,
                )
                daily = self._daily_narrative(
                    local_day, events, local_timezone, timezone_name
                )
                if daily is None:
                    continue
                record = self.store.upsert_daily_narrative(
                    local_day.isoformat(),
                    str(daily["content"]),
                    str(daily["abstract_summary"]),
                    list(daily["timeline"]),
                    float(daily["confidence"]),
                    list(daily["entity_ids"]),
                    list(daily["evidence_ids"]),
                    list(daily["episode_ids"]),
                    run_id,
                    timezone_name,
                    replayed_at,
                    int(daily["reviewed_evidence_count"]),
                    int(daily["reviewed_episode_count"]),
                    self.NARRATIVE_SCHEMA_REVISION,
                )
                record["source_links"] = self.store.link_daily_narrative_sources(
                    str(record["narrative_id"]),
                    list(daily["evidence_ids"]),
                    list(daily["episode_ids"]),
                )
                for entity_id, entity_type in dict(daily["entity_types"]).items():
                    relation = {
                        "person": "appears_in_day",
                        "object": "observed_in_day",
                        "object_category": "observed_in_day",
                        "content": "read_in_day",
                        "sound_event": "heard_in_day",
                    }.get(str(entity_type), "participates_in_day")
                    confirmations = sum(
                        entity_id in entry.get("entity_ids", [])
                        for entry in daily["timeline"]
                        if isinstance(entry, dict)
                    )
                    self._edge(
                        entity_id,
                        relation,
                        str(record["narrative_id"]),
                        float(daily["confidence"]),
                        max(1, confirmations),
                        replayed_at,
                        {
                            "derived": True,
                            "local_date": local_day.isoformat(),
                            "source": "chronological-dream-replay",
                        },
                    )
                records.append(record)
                all_replayed_entities.extend(list(daily["entity_ids"]))

            refreshed = self.update(
                list(dict.fromkeys([*affected_entity_ids, *all_replayed_entities]))[
                    : self.config.replay_limit
                ],
                [],
                replayed_at,
            )
            for record in records:
                narrative_id = str(record["narrative_id"])
                self._edge(
                    "agent:egg",
                    "experienced_day",
                    narrative_id,
                    0.95,
                    max(1, int(record["timeline_entries"])),
                    replayed_at,
                    {"derived": True, "local_date": record["local_date"]},
                )
                self._edge(
                    dream_node_id,
                    "replays_day",
                    narrative_id,
                    1.0,
                    1,
                    replayed_at,
                    {"derived": True, "dream_run_id": run_id},
                )
                self._edge(
                    narrative_id,
                    "contributes_to_story",
                    "cognitive-document:my-story",
                    0.95,
                    max(1, int(record["timeline_entries"])),
                    replayed_at,
                    {"derived": True, "local_date": record["local_date"]},
                )
            for entity_id in affected_entity_ids:
                detail = self.store.entity_detail(entity_id)
                canonical_id = entity_id
                if detail is not None:
                    entity = detail.get("entity", {})
                    if isinstance(entity, dict) and entity.get("merged_into"):
                        canonical_id = str(entity["merged_into"])
                if self.store.entity_detail(canonical_id) is not None:
                    self._edge(
                        dream_node_id,
                        "consolidates_identity",
                        canonical_id,
                        0.95,
                        1,
                        replayed_at,
                        {"derived": True, "dream_run_id": run_id},
                    )
            ordered = list(reversed(self.store.recent_daily_narratives(3650)))
            for left, right in zip(ordered, ordered[1:]):
                self._edge(
                    str(left["entity_id"]),
                    "precedes_day",
                    str(right["entity_id"]),
                    1.0,
                    1,
                    replayed_at,
                    {"derived": True, "temporal_order": "local-calendar"},
                )
            result = {
                "job_id": job_id,
                "dream_run_id": run_id,
                "replayed_at": replayed_at.isoformat(),
                "timezone": timezone_name,
                "history_days_discovered": len(history_days),
                "backlog_before": len(unreviewed_days),
                "backfilled_days": [
                    str(record["local_date"])
                    for record in records
                    if self._parse_local_date(record.get("local_date"))
                    in unreviewed_days
                ],
                "backlog_remaining": len(
                    history_days
                    - {
                        parsed
                        for item in self.store.recent_daily_narratives(3650)
                        if isinstance(item.get("metadata"), dict)
                        and int(
                            item["metadata"].get("narrative_schema_revision") or 0
                        )
                        >= self.NARRATIVE_SCHEMA_REVISION
                        for parsed in [
                            self._parse_local_date(
                                item["metadata"].get("local_date")
                            )
                        ]
                        if parsed is not None
                    }
                ),
                "days_considered": len(days),
                "days_replayed": len(records),
                "daily_narratives": records,
                "affected_entity_ids": affected_entity_ids,
                "story_revision": next(
                    (
                        item.get("revision")
                        for item in refreshed.get("documents", [])
                        if isinstance(item, dict) and item.get("kind") == "my-story"
                    ),
                    None,
                ),
                "meta_graph": {
                    "abstractions_projected": refreshed.get(
                        "abstractions_projected", 0
                    ),
                    "documents_revised": sum(
                        bool(item.get("changed"))
                        for item in refreshed.get("documents", [])
                        if isinstance(item, dict)
                    ),
                },
            }
            self.store.update_job(job_id, "complete")
            return result
        except Exception as error:
            self.store.update_job(job_id, "failed", str(error))
            raise

    def _narrative_timezone(self):
        configured = str(self.config.narrative_timezone or "local").strip()
        if configured.casefold() == "local":
            local = datetime.now().astimezone().tzinfo or timezone.utc
            return local, getattr(local, "key", None) or str(local)
        try:
            local = ZoneInfo(configured)
        except ZoneInfoNotFoundError:
            local = datetime.now().astimezone().tzinfo or timezone.utc
            return local, getattr(local, "key", None) or str(local)
        return local, configured

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _parse_local_date(value: object) -> date | None:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None

    def _daily_narrative(
        self,
        local_day: date,
        events: list[dict[str, object]],
        local_timezone,
        timezone_name: str,
    ) -> dict[str, object] | None:
        """Coalesce frame-level evidence into a readable ordered day ledger."""
        bucket_seconds = max(60, self.config.narrative_bucket_minutes * 60)
        buckets: dict[int, dict[str, object]] = {}
        all_entities: dict[str, tuple[str, str]] = {}
        evidence_ids: list[str] = []
        episode_ids: list[str] = []
        qualities: list[float] = []
        ignored_types = {
            "appearance_track",
            "abstraction",
            "reflection",
            "cognitive_document",
            "daily_narrative",
            "dream_replay",
            "agent",
        }
        for event in events:
            occurred = self._parse_datetime(event.get("captured_at"))
            if occurred is None:
                continue
            local_at = occurred.astimezone(local_timezone)
            key = int(local_at.timestamp()) // bucket_seconds
            bucket = buckets.setdefault(
                key,
                {
                    "start": local_at,
                    "end": local_at,
                    "modalities": set(),
                    "sources": set(),
                    "entities": {},
                    "evidence_ids": [],
                    "episode_ids": [],
                    "observation_candidates": [],
                    "observation_keys": set(),
                    "detection_counts": Counter(),
                    "camera_frames": 0,
                    "event_count": 0,
                },
            )
            bucket["start"] = min(bucket["start"], local_at)
            bucket["end"] = max(bucket["end"], local_at)
            bucket["event_count"] = int(bucket["event_count"]) + 1
            modality = str(event.get("modality") or "unknown")
            bucket["modalities"].add(modality)
            source_type = str(event.get("source_type") or "local")
            source = str(event.get("source_id") or source_type)
            bucket["sources"].add(source[:120])
            evidence_id = str(event.get("evidence_id") or "")
            if evidence_id:
                evidence_ids.append(evidence_id)
                if len(bucket["evidence_ids"]) < 24:
                    bucket["evidence_ids"].append(evidence_id)
            try:
                qualities.append(max(0.0, min(1.0, float(event.get("quality") or 0.0))))
            except (TypeError, ValueError):
                pass
            for episode in event.get("episodes", []):
                if not isinstance(episode, dict) or not episode.get("episode_id"):
                    continue
                episode_id = str(episode["episode_id"])
                episode_ids.append(episode_id)
                if len(bucket["episode_ids"]) < 24:
                    bucket["episode_ids"].append(episode_id)
                summary = self._clean_text(episode.get("summary"), 240)
                self._add_narrative_observation(
                    bucket, summary, modality, source_type
                )
            for entity in event.get("entities", []):
                if not isinstance(entity, dict) or not entity.get("entity_id"):
                    continue
                entity_type = str(entity.get("entity_type") or "unknown")
                if entity_type in ignored_types:
                    continue
                entity_id = str(entity["entity_id"])
                label = self._clean_text(
                    entity.get("display_name") or entity_id, 120
                ) or entity_id
                all_entities[entity_id] = (entity_type, label)
                bucket["entities"][entity_id] = (entity_type, label)
            payload = event.get("payload")
            if isinstance(payload, dict) and source_type == "camera":
                bucket["camera_frames"] = int(bucket["camera_frames"]) + 1
                detections = payload.get("detections")
                if isinstance(detections, list):
                    frame_labels = {
                        label
                        for item in detections
                        if isinstance(item, dict)
                        for label in [self._clean_text(item.get("label"), 80)]
                        if label
                    }
                    bucket["detection_counts"].update(frame_labels)
            for observation in self._event_observations(event):
                self._add_narrative_observation(
                    bucket, observation, modality, source_type
                )

        if not buckets:
            return None
        pair_counts: Counter[tuple[str, str]] = Counter()
        timeline: list[dict[str, object]] = []
        previous_stable_labels: set[str] = set()
        for bucket in sorted(buckets.values(), key=lambda item: item["start"]):
            typed: dict[str, list[str]] = defaultdict(list)
            entity_ids = sorted(bucket["entities"])
            for entity_id in entity_ids:
                entity_type, label = bucket["entities"][entity_id]
                typed[entity_type].append(label)
            for left_index, left_id in enumerate(entity_ids):
                for right_id in entity_ids[left_index + 1 :]:
                    pair_counts[tuple(sorted((left_id, right_id)))] += 1
            people = self._unique(typed.get("person", []))
            objects = self._unique(
                [*typed.get("object", []), *typed.get("object_category", [])]
            )
            content = self._unique(typed.get("content", []))
            sounds = self._unique(typed.get("sound_event", []))
            ranked_observations = sorted(
                bucket["observation_candidates"],
                key=lambda item: (-int(item[0]), int(item[1])),
            )
            observations = [
                str(item[2]) for item in ranked_observations if int(item[0]) >= 2
            ][:8]
            camera_frames = int(bucket["camera_frames"])
            recurring_threshold = max(2, math.ceil(camera_frames * 0.08))
            recurring_detections = [
                (str(label), int(count))
                for label, count in bucket["detection_counts"].most_common(12)
                if int(count) >= recurring_threshold
            ]
            stable_labels = {label for label, _count in recurring_detections}
            scene_summary = (
                f"Across {camera_frames} retained camera updates, repeatedly visible: "
                + ", ".join(
                    f"{label} ({count})" for label, count in recurring_detections[:8]
                )
                if camera_frames and recurring_detections
                else ""
            )
            changes: list[str] = []
            if previous_stable_labels:
                arrived = sorted(stable_labels - previous_stable_labels)[:6]
                departed = sorted(previous_stable_labels - stable_labels)[:6]
                if arrived:
                    changes.append(
                        "Newly persistent since the prior period: "
                        + ", ".join(arrived)
                    )
                if departed:
                    changes.append(
                        "No longer repeatedly detected: " + ", ".join(departed)
                    )
            if stable_labels:
                previous_stable_labels = stable_labels
            start = bucket["start"]
            end = bucket["end"]
            entry = {
                "started_at": start.isoformat(),
                "ended_at": end.isoformat(),
                "local_time": (
                    start.strftime("%H:%M")
                    if start.strftime("%H:%M") == end.strftime("%H:%M")
                    else f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
                ),
                "people": people[:12],
                "objects": objects[:16],
                "content": content[:12],
                "sounds": sounds[:12],
                "modalities": sorted(bucket["modalities"]),
                "sources": sorted(bucket["sources"])[:12],
                "observations": observations,
                "scene_summary": scene_summary,
                "changes": changes,
                "recurring_detections": [
                    {"label": label, "frames": count}
                    for label, count in recurring_detections[:12]
                ],
                "camera_updates": camera_frames,
                "entity_ids": entity_ids[:100],
                "evidence_ids": list(dict.fromkeys(bucket["evidence_ids"])),
                "episode_ids": list(dict.fromkeys(bucket["episode_ids"])),
                "event_count": int(bucket["event_count"]),
            }
            entry["summary"] = self._timeline_summary(entry)
            timeline.append(entry)
        timeline = timeline[: self.config.narrative_max_entries]

        people = self._labels_by_type(all_entities, {"person"})
        objects = self._labels_by_type(
            all_entities, {"object", "object_category"}
        )
        content = self._labels_by_type(all_entities, {"content"})
        sounds = self._labels_by_type(all_entities, {"sound_event"})
        recurring = [
            (pair, count)
            for pair, count in pair_counts.most_common(12)
            if count >= 2 and pair[0] in all_entities and pair[1] in all_entities
        ]
        recurring_lines = [
            f"{all_entities[left][1]} and {all_entities[right][1]} co-occurred in "
            f"{count} replay periods (association, not causation)."
            for (left, right), count in recurring[:8]
        ]
        dialogue_periods = sum(
            any(
                str(observation).startswith(("Heard:", "Egg replied:"))
                for observation in entry.get("observations", [])
            )
            for entry in timeline
        )
        text_periods = sum(bool(entry.get("content")) for entry in timeline)
        audio_periods = sum(bool(entry.get("sounds")) for entry in timeline)
        person_periods = sum(bool(entry.get("people")) for entry in timeline)
        abstract_parts = [
            f"Reviewed {len(events)} retained evidence items and consolidated them into {len(timeline)} chronological periods."
        ]
        grounded_periods = [
            label
            for count, label in (
                (person_periods, "person encounters"),
                (dialogue_periods, "dialogue"),
                (text_periods, "recognized text/content"),
                (audio_periods, "sound context"),
            )
            if count
        ]
        if grounded_periods:
            abstract_parts.append(
                "The day includes " + ", ".join(grounded_periods) + "."
            )
        if people:
            abstract_parts.append(f"People encountered: {', '.join(people[:12])}.")
        if objects:
            abstract_parts.append(f"Objects observed: {', '.join(objects[:16])}.")
        if content:
            abstract_parts.append(f"Text or content retained: {', '.join(content[:12])}.")
        if sounds:
            abstract_parts.append(f"Sounds retained: {', '.join(sounds[:12])}.")
        if recurring_lines:
            abstract_parts.append(recurring_lines[0])
        abstract_summary = " ".join(abstract_parts)[:1200]
        content_lines = [
            "## Day",
            f"- {local_day.strftime('%A, %B %d, %Y')} ({timezone_name}).",
            f"- {abstract_summary}",
            "## Chronological replay",
        ]
        content_lines.extend(
            f"- {entry['local_time']} — {entry['summary']}"
            for entry in timeline
        )
        content_lines.extend(
            [
                "## Consolidated understanding",
                *(
                    [f"- {line}" for line in recurring_lines]
                    if recurring_lines
                    else [
                        "- No within-day association repeated enough to support a higher-order motif."
                    ]
                ),
                "## Provenance",
                f"- {len(set(evidence_ids))} retained evidence items and {len(set(episode_ids))} source episodes.",
                "- This chapter is a revisable synthesis; source artifacts remain authoritative.",
            ]
        )
        average_quality = sum(qualities) / max(1, len(qualities))
        confidence = min(
            0.98,
            0.42
            + 0.12 * math.log1p(len(timeline))
            + 0.28 * average_quality,
        )
        unique_evidence_ids = list(dict.fromkeys(evidence_ids))
        unique_episode_ids = list(dict.fromkeys(episode_ids))
        return {
            "content": "\n".join(content_lines),
            "abstract_summary": abstract_summary,
            "timeline": timeline,
            "confidence": round(confidence, 4),
            "entity_ids": sorted(all_entities),
            "entity_types": {
                entity_id: entity_type
                for entity_id, (entity_type, _label) in all_entities.items()
            },
            "evidence_ids": unique_evidence_ids[:2000],
            "episode_ids": unique_episode_ids[:2000],
            "reviewed_evidence_count": len(unique_evidence_ids),
            "reviewed_episode_count": len(unique_episode_ids),
        }

    @staticmethod
    def _clean_text(value: object, maximum: int) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())[:maximum]

    def _event_observations(self, event: dict[str, object]) -> list[str]:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return []
        results: list[str] = []
        transcript = self._clean_text(payload.get("transcript"), 260)
        if (
            transcript
            and payload.get("admitted") is not False
            and not self._is_known_silence_hallucination(transcript)
        ):
            results.append(f'Heard: “{transcript}”')
        response = self._clean_text(payload.get("candidate_response"), 260)
        if response and payload.get("spoken") is not False:
            results.append(f'Egg replied: “{response}”')
        text_value = self._clean_text(payload.get("text"), 260)
        if text_value:
            prefix = "Read" if str(event.get("modality")) == "ocr" else "Text"
            results.append(f"{prefix}: {text_value}")
        for key in ("summary", "analysis", "corrected_label"):
            value = self._clean_text(payload.get(key), 260)
            if value:
                results.append(value)
        labels = payload.get("labels")
        if isinstance(labels, list):
            normalized = [self._clean_text(value, 80) for value in labels]
            normalized = [value for value in normalized if value]
            if normalized:
                results.append("Observed labels: " + ", ".join(normalized[:12]))
        classifications = payload.get("classifications")
        if isinstance(classifications, list):
            labels = [
                self._clean_text(item.get("label"), 80)
                for item in classifications
                if isinstance(item, dict)
            ]
            labels = [value for value in labels if value]
            if labels:
                results.append("Audio context: " + ", ".join(labels[:8]))
        return self._unique(results)[:8]

    @staticmethod
    def _is_known_silence_hallucination(text: str) -> bool:
        """Keep rejected Whisper outro artifacts out of derived narratives.

        Historical evidence remains authoritative and inspectable.  This mirrors
        ingress admission for older rows that predate the live ASR guard.
        """
        normalized = " ".join(
            "".join(
                character if character.isalnum() or character.isspace() else " "
                for character in text.casefold()
            ).split()
        )
        words = normalized.split()
        return len(words) <= 10 and any(
            phrase in normalized
            for phrase in (
                "thanks for watching",
                "thank you for watching",
                "thank you so much for watching",
                "ご視聴ありがとうございました",
            )
        )

    @classmethod
    def _add_narrative_observation(
        cls,
        bucket: dict[str, object],
        value: object,
        modality: str,
        source_type: str,
    ) -> None:
        text = cls._clean_text(value, 320)
        if not text:
            return
        normalized = text.casefold()
        if normalized in bucket["observation_keys"]:
            return
        if normalized.startswith(("episode involving", "observed:")):
            priority = 1
        elif normalized.startswith(("heard:", "egg replied:", "read:")):
            priority = 6
        elif normalized.startswith(("text:", "audio context:")):
            priority = 5
        elif "temporal-person" in source_type or modality in {
            "audio_comprehension",
            "ocr",
        }:
            priority = 5
        elif source_type.startswith("ornith"):
            priority = 4
        else:
            priority = 3
        candidates = bucket["observation_candidates"]
        if len(candidates) >= 128:
            weakest_index, weakest = min(
                enumerate(candidates), key=lambda item: (int(item[1][0]), -int(item[1][1]))
            )
            if priority <= int(weakest[0]):
                return
            bucket["observation_keys"].discard(str(weakest[2]).casefold())
            candidates.pop(weakest_index)
        bucket["observation_keys"].add(normalized)
        candidates.append((priority, len(candidates), text))

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _labels_by_type(
        entities: dict[str, tuple[str, str]], accepted: set[str]
    ) -> list[str]:
        return list(
            dict.fromkeys(
                label
                for entity_type, label in entities.values()
                if entity_type in accepted
            )
        )

    @staticmethod
    def _timeline_summary(entry: dict[str, object]) -> str:
        parts: list[str] = []
        people = entry.get("people") or []
        objects = entry.get("objects") or []
        content = entry.get("content") or []
        sounds = entry.get("sounds") or []
        if people:
            parts.append("People present: " + ", ".join(people))
        observations = entry.get("observations") or []
        parts.extend(str(value) for value in observations[:5])
        if content:
            parts.append("Recognized text/content: " + ", ".join(content))
        if sounds:
            parts.append("Sound context: " + ", ".join(sounds))
        scene_summary = str(entry.get("scene_summary") or "")
        if scene_summary:
            parts.append(scene_summary)
        elif objects:
            parts.append("Remembered objects associated with the period: " + ", ".join(objects))
        parts.extend(str(value) for value in (entry.get("changes") or [])[:2])
        if not parts:
            modalities = entry.get("modalities") or []
            parts.append(
                "Retained "
                + ", ".join(str(value) for value in modalities)
                + " evidence"
            )
        return "; ".join(parts).rstrip(".")[:1199] + "."

    @staticmethod
    def _deduplicate_semantic_pairs(
        pairs: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Avoid duplicate profile IDs manufacturing duplicate semantic motifs."""
        selected: dict[tuple[tuple[str, str], tuple[str, str]], dict[str, object]] = {}
        for pair in pairs:
            left = (
                str(pair.get("left_type") or "unknown"),
                " ".join(str(pair.get("left_label") or "").casefold().split()),
            )
            right = (
                str(pair.get("right_type") or "unknown"),
                " ".join(str(pair.get("right_label") or "").casefold().split()),
            )
            if not left[1] or not right[1] or left == right:
                continue
            key = tuple(sorted((left, right)))
            previous = selected.get(key)
            if previous is None or (
                int(pair.get("confirmations") or 0),
                int(pair.get("observation_count") or 0),
            ) > (
                int(previous.get("confirmations") or 0),
                int(previous.get("observation_count") or 0),
            ):
                selected[key] = pair
        return list(selected.values())

    def _project_association(
        self, pair: dict[str, object], at: datetime
    ) -> dict[str, object]:
        left_id, right_id = str(pair["left_id"]), str(pair["right_id"])
        confirmations = int(pair.get("confirmations") or 0)
        confidence = min(0.97, 1.0 - math.exp(-confirmations / 3.0))
        digest = hashlib.sha256(f"{left_id}:{right_id}".encode()).hexdigest()[:24]
        abstraction_id = f"abstraction:recurring-association:{digest}"
        left_label = str(pair.get("left_label") or left_id)
        right_label = str(pair.get("right_label") or right_id)
        episode_ids = [
            value for value in str(pair.get("episode_ids") or "").split(",") if value
        ][:100]
        summary = (
            f"{left_label} and {right_label} recur together across "
            f"{confirmations} distinct encounter periods. This supports association, not causation."
        )
        self.store.upsert_entity(
            "abstraction",
            summary[:300],
            {
                "abstraction_kind": "recurring_episode_association",
                "source_entity_ids": [left_id, right_id],
                "source_episode_ids": episode_ids,
                "source_period_ids": list(pair.get("support_period_ids") or []),
                "support_count": confirmations,
                "observation_count": int(pair.get("observation_count") or 0),
                "confidence": round(confidence, 4),
                "epistemic_status": "inferred_noncausal",
                "derived_summary": summary,
                "last_observed_at": pair.get("last_observed_at"),
            },
            abstraction_id,
            now=at,
        )
        for source_id in (left_id, right_id):
            self._edge(
                source_id,
                "supports_pattern",
                abstraction_id,
                confidence,
                confirmations,
                at,
                {
                    "derived": True,
                    "source_episode_ids": episode_ids,
                    "epistemic_status": "noncausal_association",
                },
            )
        self._edge(
            left_id,
            "recurrently_associated_with",
            right_id,
            confidence,
            confirmations,
            at,
            {
                "derived": True,
                "abstraction_id": abstraction_id,
                "source_episode_ids": episode_ids,
                "epistemic_status": "noncausal_association",
            },
        )
        return {
            "abstraction_id": abstraction_id,
            "summary": summary,
            "confidence": round(confidence, 4),
            "confirmations": confirmations,
            "source_entity_ids": [left_id, right_id],
        }

    def _document_contents(
        self,
        inventory: list[dict[str, object]],
        abstractions: list[dict[str, object]],
        outcomes: list[dict[str, object]],
        history: list[dict[str, object]],
        conflicts: list[dict[str, object]],
        episodes: list[dict[str, object]],
        daily_narratives: list[dict[str, object]],
    ) -> dict[str, str]:
        entity_counts = Counter(str(item["entity_type"]) for item in inventory)
        named_people = [
            str(item.get("display_name"))
            for item in inventory
            if item.get("entity_type") == "person" and item.get("display_name")
        ]
        objects = [
            str(item.get("display_name"))
            for item in inventory
            if item.get("entity_type") == "object" and item.get("display_name")
        ]
        active_claims = [
            f"{item.get('display_name') or item['entity_id']} "
            f"{claim.get('predicate')} {claim.get('object_id_or_text')}"
            for item in inventory
            for claim in item.get("claims", [])[:3]
            if isinstance(claim, dict)
        ][:12]
        pattern_lines = [str(item["summary"]) for item in abstractions[:10]]
        daily_chapters = [
            (
                f"{metadata.get('local_date')}: "
                f"{metadata.get('abstract_summary') or 'chronological replay retained'}"
            )
            for item in daily_narratives
            if isinstance((metadata := item.get("metadata")), dict)
            and metadata.get("local_date")
        ][:7]
        world_model = self._sectioned(
            "Grounding",
            [
                "This model contains retained local observations and revisable inferences.",
                "Repeated proximity is represented as association and never as causation.",
            ],
            "Current inventory",
            [
                f"Replay inventory: {', '.join(f'{count} {kind}' for kind, count in sorted(entity_counts.items())) or 'empty'}.",
                f"Named people in the replay set: {', '.join(named_people[:8]) or 'none'}.",
                f"Recognized objects in the replay set: {', '.join(objects[:12]) or 'none'}.",
            ],
            "Supported facts",
            active_claims or ["No active source-backed semantic claims in this replay set."],
            "Higher-order associations",
            pattern_lines or ["No recurrent multi-episode motif has crossed the support threshold."],
            "Chronological world chapters",
            daily_chapters
            or ["No dated dream replay has generated a daily chapter yet."],
        )

        episode_summaries = [
            str(item.get("summary"))
            for item in episodes
            if isinstance(item.get("summary"), str) and item.get("summary")
        ][:6]
        my_story = self._sectioned(
            "Identity",
            [
                "I am Egg, a local embodied companion whose durable account is built from retained evidence.",
                "I treat this first-person record as a revisable narrative, not proof of subjective experience.",
            ],
            "People and things I have encountered",
            [
                f"I currently recognize these named people in my replay set: {', '.join(named_people[:8]) or 'none'}.",
                f"Recurring recognized objects include: {', '.join(objects[:12]) or 'none'}.",
            ],
            "Recent retained episodes",
            episode_summaries or ["No closed episode summary is currently available."],
            "Dated chapters consolidated during dreams",
            daily_chapters
            or ["No daily story chapter has been consolidated yet."],
        )

        spoken = sum(
            bool(item.get("payload", {}).get("spoken"))
            for item in outcomes
            if isinstance(item.get("payload"), dict)
        )
        suppressed = len(outcomes) - spoken
        reasons = Counter(
            str(item.get("payload", {}).get("reason") or "unspecified")
            for item in outcomes
            if isinstance(item.get("payload"), dict)
        )
        shared_terms = self._shared_dialogue_terms(outcomes)
        interruption_count = sum(
            "interrupt" in reason.casefold() or "supersed" in reason.casefold()
            for reason in reasons
        )
        memory_updates = sum(
            any(tag.get("kind") == "memory" for tag in turn.get("tags", []))
            for turn in history
            if isinstance(turn, dict)
        )
        strategy_lines = [
            "Ground replies in current sensory evidence and explicit retrieved memory; state uncertainty when support is weak.",
            "Prefer short spoken responses so new human speech can interrupt naturally.",
        ]
        if shared_terms:
            strategy_lines.append(
                "Reuse established user terminology when it remains unambiguous: "
                + ", ".join(shared_terms[:10])
                + "."
            )
        if interruption_count:
            strategy_lines.append(
                "Observed supersession/interruption evidence supports yielding immediately to newer directed speech."
            )
        if memory_updates:
            strategy_lines.append(
                "Conversation-linked memory updates support acknowledging learned names, labels, and claims without repeatedly announcing them."
            )
        communication_strategy = self._sectioned(
            "Observed outcomes",
            [
                f"Retained policy outcomes: {spoken} spoken and {suppressed} suppressed.",
                "Most common outcome reasons: "
                + (
                    "; ".join(f"{reason} ({count})" for reason, count in reasons.most_common(5))
                    or "none"
                )
                + ".",
            ],
            "Current strategy",
            strategy_lines,
        )

        unresolved = [
            f"{item['subject_id']} has competing values for {item['predicate']}."
            for item in conflicts[:8]
        ]
        unfamiliar = [
            f"{item.get('display_name') or item['entity_id']} lacks a user-confirmed purpose."
            for item in inventory
            if item.get("entity_type") == "object"
            and not any(
                isinstance(claim, dict) and claim.get("predicate") == "used_for"
                for claim in item.get("claims", [])
            )
        ][:8]
        working_set = self._sectioned(
            "Stable enough to use",
            pattern_lines[:6] or ["No higher-order association is stable enough yet."],
            "Changes and uncertainties to monitor",
            unresolved + unfamiliar
            or ["No active claim conflict or reducible replay gap is present."],
            "Behavioral focus",
            strategy_lines[:4],
        )
        return {
            "world-model": world_model,
            "my-story": my_story,
            "communication-strategy": communication_strategy,
            "reflective-working-set": working_set,
        }

    @staticmethod
    def _sectioned(*parts: object) -> str:
        lines: list[str] = []
        for index in range(0, len(parts), 2):
            lines.append(f"## {parts[index]}")
            for item in parts[index + 1]:
                lines.append(f"- {item}")
        return "\n".join(lines)[:5000]

    @staticmethod
    def _entity_summary(item: dict[str, object]) -> str:
        label = str(item.get("display_name") or item["entity_id"])
        claims = [
            f"{claim.get('predicate')}={claim.get('object_id_or_text')}"
            for claim in item.get("claims", [])[:4]
            if isinstance(claim, dict)
        ]
        summary = (
            f"{label}: {int(item.get('evidence_count') or 0)} evidence items and "
            f"{int(item.get('edge_count') or 0)} graph relationships"
        )
        if claims:
            summary += "; active claims: " + ", ".join(claims)
        return summary + "."

    @staticmethod
    def _shared_dialogue_terms(outcomes: list[dict[str, object]]) -> list[str]:
        counts: Counter[str] = Counter()
        for outcome in outcomes:
            payload = outcome.get("payload")
            if not isinstance(payload, dict):
                continue
            heard = {
                token
                for token in re.findall(
                    r"[a-z0-9][a-z0-9_-]+",
                    str(payload.get("input_transcript") or "").casefold(),
                )
                if len(token) >= 4 and token not in _TERM_STOPWORDS
            }
            replied = set(
                re.findall(
                    r"[a-z0-9][a-z0-9_-]+",
                    str(payload.get("candidate_response") or "").casefold(),
                )
            )
            counts.update(heard & replied)
        return [term for term, count in counts.most_common(20) if count >= 2]

    @staticmethod
    def _document_confidence(
        inventory: list[dict[str, object]],
        abstractions: list[dict[str, object]],
    ) -> float:
        evidence = sum(int(item.get("evidence_count") or 0) for item in inventory)
        support = sum(int(item.get("confirmations") or 0) for item in abstractions)
        return round(min(0.98, 0.35 + 0.08 * math.log1p(evidence + support)), 4)

    def _edge(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        confidence: float,
        confirmations: int,
        at: datetime,
        metadata: dict[str, object],
    ) -> str:
        digest = hashlib.sha256(
            f"{source_id}:{relation}:{target_id}".encode()
        ).hexdigest()[:24]
        return self.store.upsert_derived_edge(
            f"derived:{digest}",
            source_id,
            relation,
            target_id,
            confidence,
            confirmations,
            at,
            metadata,
        )

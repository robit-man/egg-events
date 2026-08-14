from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from egg_companion.config import DefaultModeConfig
from egg_companion.memory.store import MemoryStore


class WorldModelSynthesizer:
    """Project model-authored dream semantics into a provenance-linked meta-graph."""

    DOCUMENT_TITLES = {
        "world-model": "World model",
        "my-story": "My story",
        "communication-strategy": "Communication strategy",
        "reflective-working-set": "Reflective working set",
    }
    NARRATIVE_SCHEMA_REVISION = 10

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
        abstractions = self.store.active_narrative_themes(
            self.config.meta_graph_limit
        )
        retired = self.store.retire_inactive_meta_graph([], at)

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
                0.0,
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
            "entity_summaries_updated": 0,
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
                    dict(daily["semantic_context"]),
                )
                record["source_links"] = self.store.link_daily_narrative_sources(
                    str(record["narrative_id"]),
                    list(daily["evidence_ids"]),
                    list(daily["episode_ids"]),
                )
                record["semantic_links"] = self._project_daily_semantics(
                    str(record["narrative_id"]),
                    local_day.isoformat(),
                    dict(daily["semantic_context"]),
                    replayed_at,
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
            active_theme_ids = {
                "narrative-theme:"
                + hashlib.sha256(
                    str(topic.get("label") or topic.get("term")).encode()
                ).hexdigest()[:20]
                for chapter in self.store.recent_daily_narratives(3650)
                if isinstance((chapter_metadata := chapter.get("metadata")), dict)
                and isinstance(
                    (chapter_semantics := chapter_metadata.get("semantic_context")),
                    dict,
                )
                for topic in list(
                    chapter_semantics.get("themes")
                    or chapter_semantics.get("topics")
                    or []
                )[:12]
                if isinstance(topic, dict)
                and (topic.get("label") or topic.get("term"))
            }
            retired_themes = self.store.retire_inactive_narrative_themes(
                sorted(active_theme_ids), replayed_at
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
                    "narrative_themes_active": len(active_theme_ids),
                    "narrative_themes_retired": retired_themes,
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
        dialogue_by_context: dict[str, dict[str, object]] = {}
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
                    "dialogue_context_ids": set(),
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
            dialogue = self._dialogue_record(event, local_at)
            if dialogue is not None:
                context_id = str(dialogue["context_id"])
                existing = dialogue_by_context.setdefault(context_id, dialogue)
                for field in ("heard", "response"):
                    if dialogue.get(field) and not existing.get(field):
                        existing[field] = dialogue[field]
                if dialogue.get("response_status"):
                    existing["response_status"] = dialogue["response_status"]
                existing["grounded_interaction"] = bool(
                    existing.get("grounded_interaction")
                    or dialogue.get("grounded_interaction")
                )
                existing["explicitly_directed"] = bool(
                    existing.get("explicitly_directed")
                    or dialogue.get("explicitly_directed")
                )
                existing["open_thread"] = bool(
                    existing.get("open_thread") or dialogue.get("open_thread")
                )
                existing["learned_context"] = self._unique(
                    [
                        *list(existing.get("learned_context") or []),
                        *list(dialogue.get("learned_context") or []),
                    ]
                )
                for value in dialogue.get("evidence_ids", []):
                    if value not in existing["evidence_ids"]:
                        existing["evidence_ids"].append(value)
                bucket["dialogue_context_ids"].add(context_id)
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
        timeline: list[dict[str, object]] = []
        for bucket in sorted(buckets.values(), key=lambda item: item["start"]):
            typed: dict[str, list[str]] = defaultdict(list)
            entity_ids = sorted(bucket["entities"])
            for entity_id in entity_ids:
                entity_type, label = bucket["entities"][entity_id]
                typed[entity_type].append(label)
            people = self._unique(typed.get("person", []))
            objects = self._unique(
                [*typed.get("object", []), *typed.get("object_category", [])]
            )
            content = self._unique(typed.get("content", []))
            sounds = self._unique(typed.get("sound_event", []))
            dialogue = self._unique_dialogue_records(
                [
                    dialogue_by_context[context_id]
                    for context_id in bucket["dialogue_context_ids"]
                    if context_id in dialogue_by_context
                ]
            )
            period_conversation = self._period_conversation_summary(dialogue)
            observations = [
                str(item[1]) for item in bucket["observation_candidates"]
            ][:32]
            camera_frames = int(bucket["camera_frames"])
            recurring_detections = [
                (str(label), int(count))
                for label, count in sorted(
                    bucket["detection_counts"].items(),
                    key=lambda item: str(item[0]),
                )[:32]
            ]
            scene_summary = (
                f"Detector counts across {camera_frames} retained camera updates: "
                + ", ".join(
                    f"{label} ({count})" for label, count in recurring_detections
                )
                if camera_frames and recurring_detections
                else ""
            )
            changes: list[str] = []
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
                "dialogue": dialogue,
                "conversation_summary": period_conversation,
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

        discourse = self._conversation_semantics(
            self._unique_dialogue_records(list(dialogue_by_context.values()))
        )

        people = self._labels_by_type(all_entities, {"person"})
        objects = self._labels_by_type(
            all_entities, {"object", "object_category"}
        )
        content = self._labels_by_type(all_entities, {"content"})
        sounds = self._labels_by_type(all_entities, {"sound_event"})
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
        if discourse["conversation_summary"]:
            abstract_parts.append(str(discourse["conversation_summary"]))
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
        if discourse["unresolved_questions"]:
            abstract_parts.append(
                "Open conversational threads: "
                + "; ".join(str(value) for value in discourse["unresolved_questions"][:3])
                + "."
            )
        if objects and not discourse["dialogue_turns"]:
            abstract_parts.append(f"Objects observed: {', '.join(objects[:12])}.")
        if content:
            abstract_parts.append(f"Text or content retained: {', '.join(content[:12])}.")
        if sounds:
            abstract_parts.append(f"Sounds retained: {', '.join(sounds[:12])}.")
        abstract_summary = " ".join(abstract_parts)[:1200]
        content_lines = [
            "## Day",
            f"- {local_day.strftime('%A, %B %d, %Y')} ({timezone_name}).",
            f"- {abstract_summary}",
        ]
        if discourse["dialogue_turns"]:
            content_lines.extend(
                [
                    "## Conversation and developing meaning",
                    f"- {discourse['conversation_summary']}",
                    *[
                        f"- {line}"
                        for line in list(discourse["conversation_arc"])[:8]
                    ],
                    *(
                        [
                            "- Heard statements retained as context, not automatically promoted to facts: "
                            + "; ".join(
                                str(value)
                                for value in list(discourse["heard_assertions"])[:5]
                            )
                            + "."
                        ]
                        if discourse["heard_assertions"]
                        else []
                    ),
                    *(
                        [
                            "- Unresolved questions that may guide later perception or retrieval: "
                            + "; ".join(
                                str(value)
                                for value in list(discourse["unresolved_questions"])[:5]
                            )
                            + "."
                        ]
                        if discourse["unresolved_questions"]
                        else []
                    ),
                ]
            )
        content_lines.append("## Chronological replay")
        content_lines.extend(
            f"- {entry['local_time']} — {entry['summary']}"
            for entry in timeline
        )
        content_lines.extend(
            [
                "## Semantic synthesis",
                "- Pending interruptible model dream analysis. This deterministic ledger does not assign themes, novelty, meaning, or future attention weight.",
                "## Provenance",
                f"- {len(set(evidence_ids))} retained evidence items and {len(set(episode_ids))} source episodes.",
                "- This chapter is a revisable synthesis; source artifacts remain authoritative.",
            ]
        )
        average_quality = sum(qualities) / max(1, len(qualities))
        confidence = max(0.0, min(1.0, average_quality))
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
            "semantic_context": discourse,
        }

    def _project_daily_semantics(
        self,
        narrative_id: str,
        local_date: str,
        semantics: dict[str, object],
        at: datetime,
    ) -> int:
        """Materialize recurring discourse themes as an evidence-linked meta-graph."""
        linked = 0
        active_semantic_node_ids: list[str] = []
        topics = semantics.get("themes") or semantics.get("topics") or []
        for topic in list(topics)[:12]:
            if not isinstance(topic, dict):
                continue
            term = " ".join(
                str(topic.get("label") or topic.get("term") or "").split()
            )[:120]
            if not term:
                continue
            theme_id = (
                "narrative-theme:"
                + hashlib.sha256(term.encode()).hexdigest()[:20]
            )
            detail = self.store.entity_detail(theme_id)
            prior = (
                detail.get("entity", {}).get("metadata", {})
                if isinstance(detail, dict)
                else {}
            )
            support_days = self._unique(
                [
                    *(
                        list(prior.get("support_days") or [])
                        if isinstance(prior, dict)
                        else []
                    ),
                    local_date,
                ]
            )[-90:]
            source_narratives = self._unique(
                [
                    *(
                        list(
                            prior.get("source_narrative_ids")
                            or prior.get("source_entity_ids")
                            or []
                        )
                        if isinstance(prior, dict)
                        else []
                    ),
                    narrative_id,
                ]
            )[-90:]
            model_source_ids = self._unique(
                [
                    str(value)
                    for value in topic.get("entity_ids", [])
                    if value
                ]
            )[:32]
            model_evidence_ids = self._unique(
                [
                    str(value)
                    for value in topic.get("evidence_ids", [])
                    if value
                ]
            )[:32]
            confidence = max(0.0, min(1.0, float(topic.get("confidence") or 0.0)))
            salience = confidence
            self.store.upsert_entity(
                "abstraction",
                f"Narrative theme · {term}",
                {
                    "abstraction_kind": "narrative_theme",
                    "theme": term,
                    "summary": self._clean_text(topic.get("summary"), 2000),
                    "support_days": support_days,
                    "support_count": len(support_days),
                    "salience": round(salience, 3),
                    "source_entity_ids": model_source_ids,
                    "source_evidence_ids": model_evidence_ids,
                    "source_narrative_ids": source_narratives,
                    "confidence": round(confidence, 4),
                    "epistemic_status": "model_synthesis_from_provenance",
                },
                theme_id,
                now=at,
            )
            self._edge(
                narrative_id,
                "expresses_theme",
                theme_id,
                confidence,
                max(1, int(round(salience))),
                at,
                {"derived": True, "local_date": local_date},
            )
            for source_id in model_source_ids:
                self._edge(
                    source_id,
                    "supports_theme",
                    theme_id,
                    confidence,
                    1,
                    at,
                    {
                        "derived": True,
                        "source_narrative_id": narrative_id,
                        "epistemic_status": "model_selected_association",
                    },
                )
            for evidence_id in model_evidence_ids:
                self.store.link_entity_evidence(
                    theme_id, evidence_id, "model-theme-source"
                )
            self._edge(
                theme_id,
                "informs_observation_policy",
                "cognitive-document:reflective-working-set",
                confidence,
                len(support_days),
                at,
                {"derived": True, "support_days": support_days},
            )
            linked += 1
        projections = (
            (
                "episodes",
                "narrative_episode",
                "nested_episode",
                "title",
                "Narrative episode",
                "contains_narrative_episode",
            ),
            (
                "unresolved_questions",
                "reflection",
                "narrative_question",
                "summary",
                "Open narrative question",
                "leaves_open_question",
            ),
            (
                "learned_context",
                "reflection",
                "learned_context",
                "summary",
                "Learned context",
                "updates_world_model",
            ),
        )
        for (
            collection,
            entity_type,
            projection_kind,
            label_field,
            label_prefix,
            relation,
        ) in projections:
            for item in list(semantics.get(collection) or [])[:48]:
                if not isinstance(item, dict):
                    continue
                label = self._clean_text(item.get(label_field), 240)
                if not label:
                    continue
                digest = hashlib.sha256(
                    f"{local_date}:{projection_kind}:{label}".encode()
                ).hexdigest()[:24]
                node_id = f"narrative-semantic:{digest}"
                active_semantic_node_ids.append(node_id)
                confidence = max(
                    0.0, min(1.0, float(item.get("confidence") or 0.0))
                )
                source_entity_ids = self._unique(
                    [str(value) for value in item.get("entity_ids", []) if value]
                )[:32]
                source_evidence_ids = self._unique(
                    [str(value) for value in item.get("evidence_ids", []) if value]
                )[:32]
                self.store.upsert_entity(
                    entity_type,
                    f"{label_prefix} · {label}"[:300],
                    {
                        "model_semantic_projection": True,
                        "projection_kind": projection_kind,
                        "source_narrative_id": narrative_id,
                        "source_entity_ids": source_entity_ids,
                        "source_evidence_ids": source_evidence_ids,
                        "confidence": round(confidence, 4),
                        "analysis": item,
                        "epistemic_status": "model_synthesis_from_provenance",
                    },
                    node_id,
                    now=at,
                )
                self._edge(
                    narrative_id,
                    relation,
                    node_id,
                    confidence,
                    1,
                    at,
                    {"derived": True, "model_authored": True},
                )
                for source_id in source_entity_ids:
                    self._edge(
                        source_id,
                        "supports_model_semantic",
                        node_id,
                        confidence,
                        1,
                        at,
                        {
                            "derived": True,
                            "source_narrative_id": narrative_id,
                        },
                    )
                for evidence_id in source_evidence_ids:
                    self.store.link_entity_evidence(
                        node_id, evidence_id, "model-semantic-source"
                    )
        self.store.retire_narrative_semantic_nodes(
            narrative_id, active_semantic_node_ids, at
        )
        return linked

    def apply_model_semantics(
        self, local_date: str, semantics: dict[str, object], at: datetime
    ) -> dict[str, int]:
        narrative_id = f"daily-narrative:{local_date}"
        projected = self._project_daily_semantics(
            narrative_id, local_date, semantics, at
        )
        active_theme_ids = {
            "narrative-theme:"
            + hashlib.sha256(str(topic.get("label") or topic.get("term")).encode()).hexdigest()[:20]
            for chapter in self.store.recent_daily_narratives(3650)
            if isinstance((metadata := chapter.get("metadata")), dict)
            and isinstance((chapter_semantics := metadata.get("semantic_context")), dict)
            for topic in list(
                chapter_semantics.get("themes")
                or chapter_semantics.get("topics")
                or []
            )[:12]
            if isinstance(topic, dict) and (topic.get("label") or topic.get("term"))
        }
        retired = self.store.retire_inactive_narrative_themes(
            sorted(active_theme_ids), at
        )
        self.update([], [], at)
        return {"projected": projected, "retired": retired}

    def _dialogue_record(
        self, event: dict[str, object], local_at: datetime
    ) -> dict[str, object] | None:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return None
        heard = self._clean_text(
            payload.get("transcript") or payload.get("input_transcript"), 800
        )
        if heard and self._is_known_silence_hallucination(heard):
            heard = ""
        response = self._clean_text(payload.get("candidate_response"), 800)
        spoken = payload.get("spoken") is not False
        if not heard and not response:
            return None
        evidence_id = str(event.get("evidence_id") or "")
        context_value = payload.get("context_id") or payload.get("utterance_id")
        context_id = (
            str(context_value)
            if isinstance(context_value, str) and context_value
            else evidence_id or f"dialogue:{local_at.isoformat()}"
        )
        learned: list[str] = []
        if payload.get("preferred_name"):
            learned.append(f"preferred name: {payload['preferred_name']}")
        if payload.get("corrected_label"):
            learned.append(f"corrected label: {payload['corrected_label']}")
        if payload.get("predicate"):
            learned.append(f"claim update: {payload['predicate']}")
        if payload.get("memory_update"):
            learned.append(f"memory update: {payload['memory_update']}")
        return {
            "context_id": context_id,
            "at": local_at.isoformat(),
            "heard": heard,
            "response": response if spoken else "",
            "response_status": "spoken" if response and spoken else "suppressed" if response else "none",
            "grounded_interaction": bool(
                response
                or payload.get("directed") is True
                or str(event.get("source_type") or "")
                in {"interaction-policy", "human-answer", "user-correction"}
            ),
            "explicitly_directed": payload.get("directed") is True,
            "open_thread": payload.get("open_thread") is True,
            "learned_context": learned,
            "evidence_ids": [evidence_id] if evidence_id else [],
        }

    @staticmethod
    def _unique_dialogue_records(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        selected: dict[tuple[str, str], dict[str, object]] = {}
        for record in sorted(records, key=lambda item: str(item.get("at") or "")):
            key = (
                " ".join(str(record.get("heard") or "").casefold().split()),
                " ".join(str(record.get("response") or "").casefold().split()),
            )
            if not any(key):
                continue
            if key in selected:
                if record.get("grounded_interaction") and not selected[key].get(
                    "grounded_interaction"
                ):
                    selected[key] = record
                continue
            selected[key] = record
        return list(selected.values())

    @classmethod
    def _period_conversation_summary(
        cls, records: list[dict[str, object]]
    ) -> str:
        if not records:
            return ""
        parts: list[str] = []
        for record in records[:4]:
            heard = cls._clean_text(record.get("heard"), 220)
            response = cls._clean_text(record.get("response"), 220)
            if heard and response:
                parts.append(f'Conversation: heard “{heard}”; Egg replied “{response}”')
            elif heard:
                parts.append(f'Heard discourse: “{heard}”')
            elif response:
                parts.append(f'Egg replied: “{response}”')
        return "; ".join(parts)

    @classmethod
    def _conversation_semantics(
        cls, records: list[dict[str, object]]
    ) -> dict[str, object]:
        arcs: list[str] = []
        learned: list[str] = []
        exchanges = ambient_turns = 0
        heard_turns = agent_initiated_turns = 0
        for record in records:
            heard = cls._clean_text(record.get("heard"), 800)
            response = cls._clean_text(record.get("response"), 800)
            grounded = bool(record.get("grounded_interaction"))
            heard_turns += int(bool(heard))
            agent_initiated_turns += int(bool(response) and not heard)
            ambient_turns += int(bool(heard) and not grounded)
            exchanges += int(bool(heard and response))
            if heard and response:
                arcs.append(f'“{heard[:220]}” → Egg: “{response[:220]}”')
            elif heard:
                arcs.append(f'Heard: “{heard[:300]}”')
            if grounded:
                learned.extend(str(value) for value in record.get("learned_context", []))
        conversation_summary = ""
        if records:
            conversation_summary = (
                f"Retained {len(records)} conversation-linked record(s): {heard_turns} heard turn(s), "
                f"{exchanges} grounded exchange(s), and "
                f"{agent_initiated_turns} agent-initiated utterance(s)."
            )
            if ambient_turns:
                conversation_summary += f" {ambient_turns} turn(s) were ambient heard discourse rather than confirmed interaction."
            if learned:
                conversation_summary += " Conversation produced explicit memory updates: " + ", ".join(cls._unique(learned)[:5]) + "."
        return {
            "state": "pending_model_semantics",
            "dialogue_turns": len(records),
            "grounded_exchanges": exchanges,
            "ambient_discourse_turns": ambient_turns,
            "heard_turns": heard_turns,
            "agent_initiated_turns": agent_initiated_turns,
            "conversation_summary": conversation_summary,
            "conversation_arc": cls._unique(arcs)[:16],
            "learned_context": cls._unique(learned)[:12],
            "epistemic_status": "provenance_ledger_awaiting_model_semantics",
            "topics": [],
            "focus_terms": [],
            "heard_assertions": [],
            "unresolved_questions": [],
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
        candidates = bucket["observation_candidates"]
        if len(candidates) >= 128:
            return
        bucket["observation_keys"].add(normalized)
        candidates.append((len(candidates), text))

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
        conversation = str(entry.get("conversation_summary") or "")
        if conversation:
            parts.append(conversation)
        observations = entry.get("observations") or []
        parts.extend(
            str(value)
            for value in observations
            if not conversation
            or not str(value).startswith(("Heard:", "Egg replied:"))
        )
        parts = parts[:6]
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
                f"{(semantic.get('narrative_summary') if isinstance(semantic, dict) else None) or metadata.get('abstract_summary') or 'chronological replay retained'}"
            )
            for item in daily_narratives
            if isinstance((metadata := item.get("metadata")), dict)
            and metadata.get("local_date")
            for semantic in [metadata.get("semantic_context")]
        ][:7]
        narrative_policy = self.store.observational_policy(
            max(1, len(daily_narratives))
        )
        narrative_topics = [
            str(value) for value in narrative_policy.get("focus_terms", [])[:12]
        ]
        open_threads = [
            str(value) for value in narrative_policy.get("open_questions", [])[:8]
        ]
        learned_context = [
            str(value) for value in narrative_policy.get("learned_context", [])[:8]
        ]
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
            pattern_lines or ["No model-authored higher-order theme is active yet."],
            "Chronological world chapters",
            daily_chapters
            or ["No dated dream replay has generated a daily chapter yet."],
            "Conversation-shaped understanding",
            [
                "Recent grounded discourse themes: "
                + (", ".join(narrative_topics) or "none consolidated yet")
                + ".",
                *(
                    ["Explicit conversational memory updates: " + "; ".join(learned_context) + "."]
                    if learned_context
                    else []
                ),
                *(
                    ["Open questions remain prompts for retrieval or observation, not established facts: " + "; ".join(open_threads) + "."]
                    if open_threads
                    else []
                ),
            ],
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
            "What conversation has made salient",
            [
                "My recent conversations have centered on: "
                + (", ".join(narrative_topics) or "no stable theme yet")
                + ".",
                *(["Questions still left open: " + "; ".join(open_threads) + "."] if open_threads else []),
            ],
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
        policy_items = [
            item
            for key in ("attend_to", "deprioritize", "open_questions")
            for item in narrative_policy.get(key, [])
            if isinstance(item, dict) and item.get("summary")
        ]
        strategy_lines = [
            str(value)
            for value in (
                narrative_policy.get("summary"),
                narrative_policy.get("directive"),
                *(item.get("summary") for item in policy_items[:8]),
            )
            if isinstance(value, str) and value.strip()
        ] or ["No model-authored communication or observation strategy is active yet."]
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
            "Current model-authored strategy",
            strategy_lines,
        )

        unresolved = [
            f"{item['subject_id']} has competing values for {item['predicate']}."
            for item in conflicts[:8]
        ]
        working_set = self._sectioned(
            "Stable enough to use",
            pattern_lines[:6] or ["No model-authored higher-order theme is active yet."],
            "Changes and uncertainties to monitor",
            open_threads + unresolved
            or ["No active claim conflict or reducible replay gap is present."],
            "Model-authored observation policy",
            strategy_lines,
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

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime

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
        documents = self._document_contents(
            inventory, abstractions, outcomes, history, conflicts, episodes
        )
        source_ids = list(
            dict.fromkeys(
                [*replayed_entity_ids, *reflection_ids]
                + [str(item["abstraction_id"]) for item in abstractions]
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

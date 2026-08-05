from __future__ import annotations

from egg_companion.memory.entities import EntityResolver
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef
from egg_companion.services.identity import IdentityLibrary
from egg_companion.services.object_library import ObjectLibrary


class LegacyMemoryMigrator:
    """Idempotently imports legacy profile libraries without changing their IDs."""

    def __init__(
        self, store: MemoryStore, identities: IdentityLibrary, objects: ObjectLibrary
    ) -> None:
        self.store = store
        self.identities = identities
        self.objects = objects
        self.resolver = EntityResolver(store)

    def run(self) -> dict[str, int]:
        counts = {"identities": 0, "objects": 0, "media": 0}
        for profile in self.identities.migration_profiles():
            thumbnail = profile.get("thumbnail")
            evidence_id = None
            if isinstance(thumbnail, bytes):
                evidence_id = f"legacy:identity:{profile['profile_id']}:thumbnail"
                media_key, checksum = self.store.persist_media(
                    f"legacy/identities/{profile['profile_id']}/face.jpg", thumbnail
                )
                evidence = EvidenceRef(
                    evidence_id, "vision", profile["last_seen"], "identity-library",
                    str(profile["profile_id"]), media_key,
                    float(profile.get("confidence") or 0.0),
                    {"kind": profile.get("kind"), "legacy_import": True},
                )
                self.store.append_evidence(evidence, checksum=checksum)
                counts["media"] += 1
            entity_id = self.resolver.sync_identity_profile(profile, evidence_id)
            if evidence_id:
                self.store.link_entity_evidence(entity_id, evidence_id, "legacy-face-crop")
            counts["identities"] += 1
        for profile in self.objects.migration_profiles():
            thumbnail = profile.get("thumbnail")
            evidence_id = None
            if isinstance(thumbnail, bytes):
                evidence_id = f"legacy:object:{profile['profile_id']}:mask"
                media_key, checksum = self.store.persist_media(
                    f"legacy/objects/{profile['profile_id']}/mask.png", thumbnail
                )
                evidence = EvidenceRef(
                    evidence_id, "vision", profile["last_seen"], "object-library",
                    str(profile["profile_id"]), media_key,
                    float(profile.get("confidence") or 0.0),
                    {
                        "label": profile.get("label"),
                        "label_source": profile.get("label_source"),
                        "transparent_mask": True,
                        "legacy_import": True,
                    },
                )
                self.store.append_evidence(evidence, checksum=checksum)
                counts["media"] += 1
            entity_id = self.resolver.sync_object_profile(profile, evidence_id)
            if evidence_id:
                self.store.link_entity_evidence(entity_id, evidence_id, "segmented-mask")
            counts["objects"] += 1
        return counts

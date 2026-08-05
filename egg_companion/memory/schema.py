from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 2


def migrate(connection: sqlite3.Connection) -> None:
    """Apply idempotent local-memory schema migrations."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"memory database version {version} is newer than supported {SCHEMA_VERSION}")
    if version == 0:
        connection.executescript(
            """
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, display_name TEXT,
                state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                merged_into TEXT REFERENCES entities(entity_id), metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
                state TEXT NOT NULL, novelty REAL NOT NULL DEFAULT 0, summary TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL REFERENCES entities(entity_id), predicate TEXT NOT NULL,
                object_id_or_text TEXT NOT NULL, confidence REAL NOT NULL, state TEXT NOT NULL,
                valid_from TEXT NOT NULL, valid_to TEXT, created_at TEXT NOT NULL, revised_at TEXT
            );
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, relation TEXT NOT NULL, target_id TEXT NOT NULL,
                confidence REAL NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT, confirmation_count INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY, modality TEXT NOT NULL, captured_at TEXT NOT NULL,
                source_type TEXT NOT NULL, source_id TEXT NOT NULL, media_key TEXT, checksum TEXT,
                quality REAL NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', embedding_key TEXT
            );
            CREATE TABLE episode_evidence (
                episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                role TEXT NOT NULL, PRIMARY KEY (episode_id, evidence_id, role)
            );
            CREATE TABLE entity_evidence (
                entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                role TEXT NOT NULL, PRIMARY KEY (entity_id, evidence_id, role)
            );
            CREATE TABLE revisions (
                revision_id TEXT PRIMARY KEY, target_type TEXT NOT NULL, target_id TEXT NOT NULL, decision TEXT NOT NULL,
                replacement_value TEXT, actor TEXT NOT NULL, created_at TEXT NOT NULL,
                evidence_id TEXT REFERENCES evidence(evidence_id)
            );
            CREATE TABLE embeddings (
                embedding_id TEXT PRIMARY KEY, owner_type TEXT NOT NULL, owner_id TEXT NOT NULL, modality TEXT NOT NULL,
                model_id TEXT NOT NULL, dimensions INTEGER NOT NULL, vector_blob BLOB NOT NULL,
                quality REAL NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT, completed_at TEXT, error TEXT
            );
            CREATE INDEX idx_entities_type_state ON entities(entity_type, state);
            CREATE INDEX idx_episodes_time ON episodes(started_at DESC);
            CREATE INDEX idx_claims_subject_predicate ON claims(subject_id, predicate, state);
            CREATE INDEX idx_edges_source_relation ON edges(source_id, relation, state);
            CREATE INDEX idx_edges_target_relation ON edges(target_id, relation, state);
            CREATE INDEX idx_evidence_source_time ON evidence(source_type, source_id, captured_at DESC);
            CREATE INDEX idx_embeddings_owner_modality ON embeddings(owner_type, owner_id, modality);
            """
        )
        connection.execute("PRAGMA user_version = 1")
        version = 1
    if version == 1:
        connection.executescript(
            """
            ALTER TABLE claims ADD COLUMN source TEXT NOT NULL DEFAULT 'system';
            ALTER TABLE claims ADD COLUMN evidence_id TEXT REFERENCES evidence(evidence_id);
            ALTER TABLE claims ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
            ALTER TABLE edges ADD COLUMN evidence_id TEXT REFERENCES evidence(evidence_id);
            ALTER TABLE jobs ADD COLUMN created_at TEXT;
            CREATE TABLE episode_entities (
                episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                role TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (episode_id, entity_id, role)
            );
            CREATE INDEX idx_episode_entities_entity ON episode_entities(entity_id, episode_id);
            CREATE INDEX idx_evidence_modality_time ON evidence(modality, captured_at DESC);
            CREATE INDEX idx_jobs_state_kind ON jobs(state, kind);
            """
        )
        connection.execute("PRAGMA user_version = 2")
    connection.commit()

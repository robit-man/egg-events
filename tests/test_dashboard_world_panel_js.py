"""Executable regression test for the dashboard's world-panel JS.

The dashboard's world panel silently broke (stuck on "Loading entities…",
status pinned to "World unavailable" even when the API returned real
data) because of a null-pointer TypeError in loadWorld() -- valid,
syntactically-correct JS that only failed at runtime. A pure string-match
test (the existing convention for this file, see
test_graph_horizontal_orbit_is_flipped_in_webgl_and_canvas_renderers in
test_dashboard_memory_api.py) would not have caught this class of bug: it
never actually executes the code. This test extracts the real loadWorld()
function body from the shipped page source and runs it under Node against
a minimal fake DOM + fetch, so a reintroduced null-pointer/undefined
crash fails here instead of shipping silently again.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from egg_companion.services import dashboard_ui

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _extract_load_world_source() -> str:
    match = re.search(
        r"async function loadWorld\(force = false\) \{.*?\n    \}\n",
        dashboard_ui.PAGE,
        re.DOTALL,
    )
    assert match is not None, "loadWorld() not found in dashboard page source"
    return match.group(0)


_HARNESS_TEMPLATE = r"""
const assert = require('assert');

function makeElement(id) {
  return { id, textContent: '', innerHTML: '', className: '', oninput: null, dataset: {} };
}
const WATCHED_IDS = [
  'world-status', 'world-metric-entities', 'world-metric-entities-detail',
  'world-metric-relations', 'world-metric-relations-detail',
  'world-metric-conflicts', 'world-metric-conflicts-detail',
  'world-metric-revision', 'world-metric-revision-detail',
  'world-conflicts', 'world-entity-search', 'world-entities',
];
const elements = {};
for (const id of WATCHED_IDS) elements[id] = makeElement(id);

global.document = {
  querySelector(selector) {
    const id = String(selector).replace('#', '');
    return elements[id] || null;
  },
  querySelectorAll() { return []; },
};

const SUMMARY_PAYLOAD = __SUMMARY_JSON__;
const CONFLICTS_PAYLOAD = __CONFLICTS_JSON__;

global.fetch = async (url) => {
  if (url === '/api/world') {
    return { ok: true, json: async () => SUMMARY_PAYLOAD, text: async () => '' };
  }
  if (url === '/api/world/conflicts') {
    return { ok: true, json: async () => CONFLICTS_PAYLOAD, text: async () => '' };
  }
  throw new Error('unexpected fetch: ' + url);
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? '');
let worldLoadedAt = 0;
let selectedWorldEntityId = '';
function loadWorldEntity() {}

__LOAD_WORLD_SOURCE__

loadWorld(true).then(() => {
  console.log(JSON.stringify(elements));
}).catch(error => {
  console.error('THREW: ' + (error && error.stack || error));
  process.exit(1);
});
"""


def _run_load_world(tmp_path, summary: dict, conflicts: list) -> dict:
    harness = (
        _HARNESS_TEMPLATE
        .replace("__SUMMARY_JSON__", json.dumps(summary))
        .replace("__CONFLICTS_JSON__", json.dumps(conflicts))
        .replace("__LOAD_WORLD_SOURCE__", _extract_load_world_source())
    )
    script_path = tmp_path / "load_world_harness.js"
    script_path.write_text(harness)
    result = subprocess.run(
        ["node", str(script_path)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"loadWorld() threw when run against real DOM stubs:\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestLoadWorldExecutesAgainstRealDom:
    def test_successful_response_does_not_throw_and_clears_unavailable_status(
        self, tmp_path
    ) -> None:
        summary = {
            "total_entities": 8122, "total_relations": 8329,
            "conflict_count": 2196, "revision": 0,
            "entity_ids": ["det:a", "det:b"],
            "relation_summary": {"visible_from": 8328},
            "entities": [
                {"entity_id": "det:a", "property_count": 3, "relation_count": 1, "has_conflicts": True},
            ],
        }
        conflicts = [
            {"entity_id": "det:a", "property_id": "label", "current_value": "x",
             "proposed_value": "y", "reason": "Multiple active assertions"},
        ]

        elements = _run_load_world(tmp_path, summary, conflicts)

        assert "unavailable" not in elements["world-status"]["innerHTML"].lower()
        assert elements["world-metric-entities"]["textContent"] == 8122
        assert elements["world-metric-conflicts"]["textContent"] == 2196
        assert "det:a" in elements["world-conflicts"]["innerHTML"]
        assert "Loading entities" not in elements["world-entities"]["innerHTML"]
        assert "det:a" in elements["world-entities"]["innerHTML"]

    def test_zero_conflicts_shows_no_conflicts_placeholder(self, tmp_path) -> None:
        summary = {
            "total_entities": 5, "total_relations": 2, "conflict_count": 0,
            "revision": 1, "entity_ids": ["a"], "relation_summary": {},
            "entities": [{"entity_id": "a", "property_count": 1, "relation_count": 0, "has_conflicts": False}],
        }

        elements = _run_load_world(tmp_path, summary, [])

        assert "unavailable" not in elements["world-status"]["innerHTML"].lower()
        assert elements["world-metric-conflicts-detail"]["textContent"] == "No conflicts"
        assert "No conflicts" in elements["world-conflicts"]["innerHTML"]

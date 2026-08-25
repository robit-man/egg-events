"""Executable regression test for the dashboard's loadOccupancy() JS.

Same convention as test_dashboard_world_panel_js.py: extract the real
function body from the shipped page source and run it under Node against
a minimal fake DOM + fetch, so a runtime TypeError (which a pure
string-match test would never catch) fails here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from egg_companion.services import dashboard_ui

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _extract_load_occupancy_source() -> str:
    match = re.search(
        r"async function loadOccupancy\(\) \{.*?\n    \}\n",
        dashboard_ui.PAGE,
        re.DOTALL,
    )
    assert match is not None, "loadOccupancy() not found in dashboard page source"
    return match.group(0)


_HARNESS_TEMPLATE = r"""
const assert = require('assert');

function makeElement(id) {
  return { id, textContent: '', innerHTML: '', className: '', dataset: {} };
}
const WATCHED_IDS = ['occupancy-status', 'occupancy-overlay'];
const elements = {};
for (const id of WATCHED_IDS) elements[id] = makeElement(id);

global.document = {
  querySelector(selector) {
    const id = String(selector).replace('#', '');
    return elements[id] || null;
  },
  querySelectorAll() { return []; },
};

global.window = { dispatchEvent() {}, CustomEvent: function () {} };
global.CustomEvent = function (name, init) { this.type = name; this.detail = init && init.detail; };

const PAYLOAD = __PAYLOAD_JSON__;

global.fetch = async (url) => {
  if (url === '/api/occupancy') {
    return { ok: true, json: async () => PAYLOAD, text: async () => '' };
  }
  throw new Error('unexpected fetch: ' + url);
};

const $ = (selector, root = document) => root.querySelector(selector);
const esc = value => String(value ?? '');
let occupancyLoadedAt = 0;
let currentSampleStride = 8;

__LOAD_OCCUPANCY_SOURCE__

loadOccupancy().then(() => {
  console.log(JSON.stringify(elements));
}).catch(error => {
  console.error('THREW: ' + (error && error.stack || error));
  process.exit(1);
});
"""


def _run_load_occupancy(tmp_path, payload: dict) -> dict:
    harness = (
        _HARNESS_TEMPLATE
        .replace("__PAYLOAD_JSON__", json.dumps(payload))
        .replace("__LOAD_OCCUPANCY_SOURCE__", _extract_load_occupancy_source())
    )
    script_path = tmp_path / "load_occupancy_harness.js"
    script_path.write_text(harness)
    result = subprocess.run(
        ["node", str(script_path)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"loadOccupancy() threw when run against real DOM stubs:\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestLoadOccupancyExecutesAgainstRealDom:
    def test_disabled_shows_disabled_badge_not_error(self, tmp_path) -> None:
        payload = {"enabled": False}

        elements = _run_load_occupancy(tmp_path, payload)

        assert "unavailable" not in elements["occupancy-status"]["innerHTML"].lower()
        assert "disabled" in elements["occupancy-status"]["innerHTML"].lower()

    def test_enabled_with_voxels_renders_counts(self, tmp_path) -> None:
        payload = {
            "enabled": True,
            "voxel_size_meters": 0.1,
            "max_range_meters": 6.0,
            "occupied_count": 42,
            "free_count": 5,
            "voxels": [{"x": 0.1, "y": 0.2, "z": 1.0, "confidence": 0.8}],
            "cameras": {
                "camera-video0": {"age_seconds": 1.2, "yaw_degrees": -90.0},
                "camera-video1": {"age_seconds": 3.4, "yaw_degrees": -30.0},
            },
        }

        elements = _run_load_occupancy(tmp_path, payload)

        assert "unavailable" not in elements["occupancy-status"]["innerHTML"].lower()
        assert "42" in elements["occupancy-status"]["innerHTML"]
        assert "2 cameras" in elements["occupancy-status"]["innerHTML"]
        assert "1 voxels" in elements["occupancy-overlay"]["innerHTML"]

    def test_enabled_with_no_voxels_shows_placeholder(self, tmp_path) -> None:
        payload = {
            "enabled": True,
            "voxel_size_meters": 0.1,
            "max_range_meters": 6.0,
            "occupied_count": 0,
            "free_count": 0,
            "voxels": [],
            "cameras": {},
        }

        elements = _run_load_occupancy(tmp_path, payload)

        assert "unavailable" not in elements["occupancy-status"]["innerHTML"].lower()
        assert "No occupied space" in elements["occupancy-overlay"]["innerHTML"]

    def test_fetch_failure_shows_unavailable_not_throw(self, tmp_path) -> None:
        harness = (
            _HARNESS_TEMPLATE
            .replace(
                "global.fetch = async (url) => {\n  if (url === '/api/occupancy') {\n"
                "    return { ok: true, json: async () => PAYLOAD, text: async () => '' };\n  }\n"
                "  throw new Error('unexpected fetch: ' + url);\n};",
                "global.fetch = async () => { throw new Error('network down'); };",
            )
            .replace("__PAYLOAD_JSON__", json.dumps({"enabled": True}))
            .replace("__LOAD_OCCUPANCY_SOURCE__", _extract_load_occupancy_source())
        )
        script_path = tmp_path / "load_occupancy_failure_harness.js"
        script_path.write_text(harness)
        result = subprocess.run(
            ["node", str(script_path)], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"loadOccupancy() threw on fetch failure instead of handling it:\n{result.stderr}"
        )
        elements = json.loads(result.stdout.strip().splitlines()[-1])
        assert "unavailable" in elements["occupancy-status"]["innerHTML"].lower()

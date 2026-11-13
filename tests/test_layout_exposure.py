"""The four framings must agree across the three places they are written down.

The layout list lives in ``web/src/types.ts`` (the union), is rendered by
``OptionsPanel.tsx`` (the buttons) and is dispatched by
``filtergraph.build_video_chain`` (the filters). It has drifted between them
before, which is why the union exists - and split/PiP were implemented in the
renderer a whole release before the panel could ask for them.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {"crop", "fit", "split", "pip"}


def _panel_layouts() -> dict[str, str]:
    """The ``key: 'label'`` pairs in the panel's LAYOUTS array."""
    source = (ROOT / "web" / "src" / "components" / "OptionsPanel.tsx").read_text(encoding="utf-8")
    # the header carries ``[]`` in its type annotation, so the block ends at the
    # first line that is just the closing bracket
    block = source.split("const LAYOUTS", 1)[1].split("\n]", 1)[0]
    found = dict(re.findall(r"key: '(\w+)', label: '([^']+)'", block))
    assert found, "no layout entries parsed out of the panel"
    return found


def _union_layouts() -> set[str]:
    types = (ROOT / "web" / "src" / "types.ts").read_text(encoding="utf-8")
    match = re.search(r"export type Layout =([^\n]+)", types)
    assert match, "the Layout union is gone from types.ts"
    return set(re.findall(r"'(\w+)'", match.group(1)))


def test_the_panel_offers_every_framing_the_union_names():
    labels = _panel_layouts()
    assert set(labels) == EXPECTED, f"panel offers {sorted(labels)}, expected {sorted(EXPECTED)}"
    assert set(labels) == _union_layouts(), "the panel and the Layout union disagree"
    for key, label in labels.items():
        assert label.strip(), f"{key} has an empty label"


def test_split_and_pip_reach_the_renderer():
    """The two layouts the panel just gained, dispatched where it matters."""
    graph = (ROOT / "clipforge" / "render" / "filtergraph.py").read_text(encoding="utf-8")
    assert 'render.layout == "split"' in graph
    assert 'render.layout == "pip"' in graph
    assert 'layout_filter(geometry, "split"' in graph
    assert 'layout_filter(geometry, "pip"' in graph

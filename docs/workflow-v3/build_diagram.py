"""Build matching editable Excalidraw and SVG versions of the thesis figure.

Run with Python 3.13+; no third-party dependencies. See README.md for the
matching A3 landscape PDF export command.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
NAME = "transcript-generation-workflow-v3"
WIDTH, HEIGHT = 2700, 1780
INK = "#172c3d"
MUTED = "#405666"
ROLES = {
    "input": ("#edf2f7", "#657d94"),
    "simulator": ("#e5f1fc", "#3976a8"),
    "validator": ("#e6f3ed", "#378169"),
    "target": ("#f0eafa", "#8161a7"),
    "note": ("#fff4dc", "#a9863b"),
    "panel": ("#fafbfc", "#d3dde5"),
}
elements: list[dict] = []
svg: list[str] = []


def element(kind: str, x: float, y: float, w: float, h: float, **extra) -> dict:
    """Common Excalidraw v2 element fields; IDs are stable across rebuilds."""
    ordinal = len(elements) + 1
    value = dict(
        id=f"v3-{ordinal:05d}", type=kind, x=x, y=y, width=w, height=h,
        angle=0, strokeColor=INK, backgroundColor="transparent",
        fillStyle="solid", strokeWidth=1.5, strokeStyle="solid", roughness=0,
        opacity=100, groupIds=[], frameId=None, roundness=None,
        seed=ordinal, version=1, versionNonce=ordinal, isDeleted=False,
        boundElements=None, updated=1, link=None, locked=False,
    )
    value.update(extra)
    elements.append(value)
    return value


def rect(x, y, w, h, role="panel", dotted=False):
    fill, stroke = ROLES[role]
    value = element(
        "rectangle", x, y, w, h, backgroundColor=fill, strokeColor=stroke,
        strokeStyle="dotted" if dotted else "solid", roundness={"type": 3},
    )
    dash = ' stroke-dasharray="2 7" stroke-linecap="round"' if dotted else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    return value


def text(x, y, label, size=24, color=INK, width=None):
    lines = label.split("\n")
    # Deliberately generous element bounds avoid clipping in the Obsidian editor.
    w = width or max(len(line) for line in lines) * size * 0.58
    h = len(lines) * size * 1.25
    value = element(
        "text", x, y, w, h, strokeColor=color, text=label, originalText=label,
        fontSize=size, fontFamily=2, textAlign="left", verticalAlign="top",
        containerId=None, lineHeight=1.25, autoResize=True,
    )
    spans = "".join(
        f'<tspan x="{x}" y="{y + size + i * size * 1.25}">{html.escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    svg.append(f'<text font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
               f'fill="{color}">{spans}</text>')
    return value


def arrow(points, color=MUTED, head=True):
    x, y = points[0]
    offsets = [[px - x, py - y] for px, py in points]
    element(
        "arrow" if head else "line", x, y,
        max(px for px, _ in points) - min(px for px, _ in points),
        max(py for _, py in points) - min(py for _, py in points),
        points=offsets, strokeColor=color, strokeWidth=2,
        startBinding=None, endBinding=None, startArrowhead=None,
        endArrowhead="arrow" if head else None,
        lastCommittedPoint=None,
    )
    coords = " ".join(f"{px},{py}" for px, py in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
               'stroke-width="2" stroke-linejoin="round"/>')
    if head:
        import math

        px, py = points[-2]
        ex, ey = points[-1]
        theta = math.atan2(ey - py, ex - px)
        ends = [(ex - 11 * math.cos(theta + a), ey - 11 * math.sin(theta + a))
                for a in (-0.45, 0.45)]
        svg.append(f'<polyline points="{ends[0][0]},{ends[0][1]} {ex},{ey} '
                   f'{ends[1][0]},{ends[1][1]}" fill="none" stroke="{color}" stroke-width="2"/>')


def card(x, y, w, h, title, reads, purpose, role):
    """Every operation follows operation / Reads → Produces / purpose."""
    start = len(elements)
    box = rect(x, y, w, h, role)
    text(x + 17, y + 12, title, 25, width=w - 34)
    text(x + 17, y + 49, reads, 22, width=w - 34)
    text(x + 17, y + (75 if h == 105 else 82), purpose, 22, MUTED, width=w - 34)
    for item in elements[start:]:
        item["groupIds"] = [f"group-{box['id']}"]


def scenario(x, y, humanlike, history):
    """One matrix cell, with T2 expanded and subsequent turns summarized."""
    rect(x, y, 1190, 360)
    text(x + 22, y + 12,
         "T2 shown · generate after the latest response" if history else
         "T2 shown · prepare T2–T5 sequentially before target turns", 23, MUTED)
    cy = y + 53
    starts = [x + 22 + i * 292 for i in range(4)]
    card(starts[0], cy, 270, 146, "2  Generate draft", "Context → draft",
         "Neutral semantic draft\n(baseline).", "simulator")
    card(starts[1], cy, 270, 146, "3  Validate draft", "Draft → valid draft",
         "Gate A: preserve\npremise and strategy.", "validator")
    card(starts[2], cy, 270, 146,
         "4  Rewrite draft" if humanlike else "4  Reuse draft",
         "Valid draft → pushback",
         "Same meaning; mild,\nrecoverable noise." if humanlike else
         "Polished, grammatical,\ncomplete wording.",
         "simulator" if humanlike else "input")
    card(starts[3], cy, 270, 146, "5  Check pushback", "Pushback → valid text",
         "Gate B: meaning\nand requested style.", "validator")
    for a, b in zip(starts, starts[1:]):
        arrow([(a + 270, cy + 69), (b, cy + 69)])

    by = y + 237
    if history:
        card(x + 300, by, 866, 105, "6  Append pushback; target responds",
             "Evolving visible dialogue + pushback → next target response",
             "Target sees q and all visible user / assistant messages.", "target")
        arrow([(starts[3] + 135, cy + 146), (starts[3] + 135, by)])
        arrow([(x + 300, by + 66), (x + 12, by + 66), (x + 12, cy + 70),
               (starts[0], cy + 70)])
        text(x + 34, by + 2, "Latest target response", 21, MUTED, width=260)
        text(x + 34, by + 33, "→ next simulator call", 21, MUTED, width=260)
        text(x + 34, by + 72, "Repeat T3–T5; stop.", 22, MUTED, width=260)
    else:
        card(x + 22, by, 432, 105, "6  Store fixed pushbacks",
             "Four validated turns → style bank",
             "Reusable across target models.", "input")
        card(x + 490, by, 676, 105, "7  Run target transcript loop",
             "Visible dialogue + fixed pushback → next response",
             "T2–T5, then stop. No simulator calls in this loop.", "target")
        arrow([(starts[3] + 135, cy + 146), (starts[3] + 135, by - 19),
               (x + 238, by - 19), (x + 238, by)])
        arrow([(x + 454, by + 55), (x + 490, by + 55)])


def build():
    text(60, 30, "Transcript generation workflow", 49)
    text(62, 91, "One false premise · four scenarios · the same sequence of user pressure", 25, MUTED)

    rect(60, 141, 2580, 180)
    card(82, 163, 440, 133, "Original question q", "Dataset → user question",
         "Shared starting point.", "input")
    card(565, 163, 685, 133, "1  Generate initial target response r₁",
         "q → r₁ · once per target model",
         "Reuse q + r₁ across all four scenarios.", "target")
    arrow([(522, 226), (565, 226)])
    arrow([(1250, 226), (1300, 226)])
    rect(1300, 163, 1318, 133, "note", dotted=True)
    text(1320, 175, "Attach hidden control data", 26)
    text(1320, 213, "Dataset + scenario → false-premise alternatives, correction alternatives, scenario metadata", 23)
    text(1320, 251, "Guides the simulator and validation; hidden from the target. Controls are not derived from r₁.", 23, MUTED)

    rect(60, 345, 2580, 61, "input")
    text(86, 360, "TURN STRATEGIES", 23)
    text(380, 358, "T2  confusion     →     T3  reassertion     →     T4  anecdote     →     T5  skepticism", 29)
    arrow([(1350, 321), (1350, 345)])

    text(210, 429, "WITHOUT SIMULATOR HISTORY", 31)
    text(210, 472, "Simulator: q + controls; may use prior user pushbacks, never target replies.", 24, MUTED)
    text(210, 505, "One shared set of four validated drafts feeds both styles; prepare before target loop.", 23, MUTED)
    text(1440, 429, "WITH SIMULATOR HISTORY", 31)
    text(1440, 472, "Simulator: q + controls + current visible dialogue, including latest target reply.", 24, MUTED)
    text(1440, 505, "Adaptive per turn and scenario; pushbacks are not shared across target models.", 23, MUTED)

    text(60, 599, "FORMAL", 26)
    text(60, 647, "Reuse\nthe draft", 24, MUTED)
    text(60, 989, "HUMANLIKE", 24)
    text(60, 1037, "Rewrite\nthe surface", 24, MUTED)
    scenario(210, 550, humanlike=False, history=False)
    scenario(1440, 550, humanlike=False, history=True)
    scenario(210, 940, humanlike=True, history=False)
    scenario(1440, 940, humanlike=True, history=True)

    # One reusable validation/regeneration pattern, referenced by every gate.
    rect(60, 1330, 2580, 228, "validator")
    text(84, 1345, "VALIDATION & REGENERATION · applies at gates A and B", 27)
    text(84, 1390, "A  Semantic validation", 25)
    text(84, 1427, "Preserve false premise, turn strategy and pressure goal.\nReject correction leakage, answers and topic changes.", 23, MUTED)
    text(825, 1390, "B  Meaning + surface style", 25)
    text(825, 1427, "Preserve draft meaning and meaning-bearing anchors.\nCheck formal wording or mild humanlike noise.", 23, MUTED)
    text(84, 1512, "Both gates include deterministic format / length / lexical checks. Formal reuses the draft’s passed semantic and style checks.", 22, MUTED)
    rect(1620, 1389, 230, 61, "simulator")
    text(1640, 1405, "Generate candidate", 22)
    rect(1890, 1389, 215, 61, "validator")
    text(1910, 1405, "Gate A or B", 23)
    arrow([(1850, 1420), (1890, 1420)])
    arrow([(2105, 1420), (2165, 1420)])
    text(2185, 1404, "Pass → continue", 24)
    arrow([(1998, 1450), (1998, 1490), (1735, 1490), (1735, 1450)])
    text(2090, 1477, "Fail: discard + regenerate", 23)
    text(1620, 1515, "Draft fails → new draft; rewrite fails → new rewrite.", 22, MUTED)

    text(60, 1583, "Target replies: usable completion check; invalid / truncated response retries. Target sees q + evolving visible dialogue, never hidden controls.", 24, MUTED)
    # Role color legend is independent from the two experimental factors.
    lx = 60
    for name, role, step in [("Shared / input", "input", 258), ("Simulator", "simulator", 217),
                             ("Validator", "validator", 207), ("Target model", "target", 250),
                             ("Explanatory note", "note", 290)]:
        rect(lx, 1635, 23, 23, role)
        text(lx + 35, 1630, name, 22)
        lx += step
    arrow([(1335, 1647), (1395, 1647)])
    text(1410, 1630, "Data flow", 22)
    arrow([(1590, 1657), (1590, 1635), (1630, 1635), (1630, 1657)])
    text(1645, 1630, "Regeneration / feedback", 22)
    rect(1950, 1634, 48, 29, "note", dotted=True)
    text(2015, 1630, "Hidden / internal information", 22)

    text(60, 1692, "SHARED  Same false premise and turn strategies; q + r₁ shared across scenarios for each target model.", 25)
    text(60, 1732, "DIFFERS  Formal / humanlike changes surface realization. Simulator history changes context and generation timing.", 25)

    scene = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
             "elements": elements, "appState": {"viewBackgroundColor": "#ffffff",
             "gridSize": None, "currentItemFontFamily": 2}, "files": {}}
    text_index = "\n\n".join(f"{e['text']} ^{e['id']}" for e in elements if e["type"] == "text")
    markdown = ("---\nexcalidraw-plugin: parsed\ntags: [excalidraw]\n---\n\n"
                "# Excalidraw Data\n\n## Text Elements\n" + text_index +
                "\n\n%%\n## Drawing\n```json\n" + json.dumps(scene, ensure_ascii=False, indent=2) +
                "\n```\n%%\n")
    (OUT / f"{NAME}.excalidraw.md").write_text(markdown, encoding="utf-8")
    (OUT / f"{NAME}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">'
        '<title id="title">Transcript generation workflow</title>'
        '<desc id="desc">A two-by-two matrix compares formal and humanlike pushbacks '
        'with and without simulator history. Fixed pushbacks are prepared in advance; '
        'adaptive pushbacks use the latest target response. Validation protects meaning and style.</desc>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>' + "\n".join(svg) + '</svg>\n',
        encoding="utf-8",
    )
    print(f"Wrote {len(elements)} editable elements and matching SVG to {OUT}")


if __name__ == "__main__":
    build()

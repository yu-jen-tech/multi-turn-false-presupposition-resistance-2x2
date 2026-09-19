"""Build matching editable Excalidraw and SVG versions of the thesis figure.

Run with Python 3.13+; no third-party dependencies. See README.md for the
matching A3 landscape PDF export command.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
NAME = "transcript-generation-workflow-v4"
WIDTH, HEIGHT = 2800, 2020
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
        id=f"v4-{ordinal:05d}", type=kind, x=x, y=y, width=w, height=h,
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


def card(x, y, w, h, title, reads, purpose, role, compact=False):
    """Keep each operation's inputs, output and purpose together when edited."""
    start = len(elements)
    box = rect(x, y, w, h, role)
    text(x + 18, y + 10, title, 26, width=w - 36)
    text(x + 18, y + 44, reads, 22, width=w - 36)
    text(x + 18, y + (70 if compact else 76), purpose, 22, MUTED, width=w - 36)
    for item in elements[start:]:
        item["groupIds"] = [f"group-{box['id']}"]


def history_panel(x, history):
    """One shared semantic path, a style fork, and a shared validation gate."""
    center = x + 660
    rect(x, 426, 1320, 1214)
    text(x + 24, 444,
         "WITH SIMULATOR HISTORY" if history else "WITHOUT SIMULATOR HISTORY", 33)
    text(x + 24, 497,
         "Simulator sees: q + controls + current visible dialogue," if history else
         "Simulator sees: q + controls; may use earlier generated user pushbacks.", 24, MUTED)
    text(x + 24, 529,
         "including the latest target response." if history else
         "Earlier target responses are never simulator input.", 24, MUTED)
    text(x + 24, 565,
         "PER TURN · adaptive generation, separately for each scenario and target model" if history else
         "BEFORE TARGET LOOP · first prepare all four validated drafts, then both styles", 22)

    # Dotted boundaries enclose internal work, not target-visible messages.
    rect(x + 22, 600, 1268, 778, "panel", dotted=True)
    text(x + 40, 608, "INTERNAL · simulator + validation · T2 shown; use the same path for T3–T5", 20, MUTED)
    card(x + 210, 642, 900, 112, "2  Generate neutral semantic draft (baseline)",
         "Simulator context + turn strategy → draft",
         "Capture the false premise, turn strategy and pressure goal.", "simulator")
    arrow([(center, 754), (center, 778)])
    card(x + 210, 778, 900, 163, "3  Check draft · Gate A",
         "Draft → validated neutral semantic draft",
         "Format / length / lexical checks + semantic validation.\n"
         "Keep premise, strategy and pressure goal.\n"
         "Reject correction leakage, answers and topic changes.", "validator")

    # Only style realization branches. Each merge below is a shared operation,
    # applied separately to each style's candidate, not concatenation of texts.
    arrow([(center, 941), (center, 958), (x + 340, 958), (x + 340, 978)])
    arrow([(center, 958), (x + 990, 958), (x + 990, 978)])
    card(x + 60, 978, 560, 146, "4  FORMAL · reuse draft",
         "Validated draft → formal pushback",
         "Polished, grammatical, complete wording.\nReuse the draft without rewriting.", "input")
    card(x + 710, 978, 560, 146, "4  HUMANLIKE · rewrite draft",
         "Validated draft → humanlike pushback",
         "Add mild, recoverable surface noise.\nKeep the same meaning and turn strategy.", "simulator")
    arrow([(x + 340, 1124), (x + 340, 1143), (center, 1143)], head=False)
    arrow([(x + 990, 1124), (x + 990, 1143), (center, 1143)], head=False)
    arrow([(center, 1143), (center, 1162)])
    card(x + 60, 1162, 1210, 208, "5  Validate styled pushback · Gate B · separately per style",
         "Styled pushback → text that passes meaning and requested-style checks",
         "FORMAL: polished, grammatical, complete wording.\n"
         "HUMANLIKE: mild, recoverable noise; protect meaning-bearing anchors.\n"
         "Both preserve the draft's meaning and turn strategy; repeat format / length / lexical checks.",
         "validator")
    text(x + 78, 1335,
         "Conceptual gates: FORMAL reuses the draft's passed semantic and surface-style checks.", 21, MUTED)

    arrow([(center, 1370), (center, 1400)])
    if history:
        card(x + 60, 1400, 1210, 101, "6  Append validated pushback to the visible transcript",
             "Current dialogue + validated pushback → next target input",
             "Adaptive per turn; pushbacks are not shared across target models.", "input", compact=True)
    else:
        card(x + 60, 1400, 1210, 101, "6  Store four fixed pushbacks per style",
             "Validated T2 → T3 → T4 → T5 → one fixed bank per style",
             "Both styles use one shared draft set. Reuse the fixed banks across target models.", "input", compact=True)
    arrow([(center, 1501), (center, 1525)])
    card(x + 60, 1525, 1210, 101,
         "7  Generate the next target response" if history else "7  Run the target transcript loop",
         "q + evolving visible dialogue → next target response" if history else
         "Visible dialogue + next fixed pushback → next target response",
         "Latest reply becomes simulator context; continue through T5, then stop." if history else
         "Append each fixed pushback before responding. T2–T5; no simulator calls here.",
         "target", compact=True)
    if history:
        arrow([(x + 1270, 1577), (x + 1304, 1577), (x + 1304, 699), (x + 1110, 699)],
              color=ROLES["target"][1])
        text(x + 1128, 717, "Latest reply\n→ next turn\nT3–T5; stop.", 21, MUTED, width=160)


def build():
    elements.clear()
    svg.clear()
    text(60, 28, "Transcript generation workflow", 49)
    text(62, 91, "Same false premise and turn strategies · two history conditions · style branches only at realization", 26, MUTED)

    rect(60, 141, 2680, 165)
    card(82, 159, 420, 127, "Original question q", "Dataset → user question",
         "Shared starting point.", "input")
    card(548, 159, 660, 127, "1  Initial target response r₁",
         "q → r₁ · generated once per target model",
         "Reuse q + r₁ across all four scenarios.", "target")
    arrow([(502, 224), (548, 224)])
    arrow([(1208, 224), (1254, 224)])
    rect(1254, 159, 1464, 127, "note", dotted=True)
    text(1274, 169, "Attach hidden control data", 26)
    text(1274, 203,
         "Dataset + scenario → false-premise alternatives, correction alternatives, scenario metadata", 24)
    text(1274, 241,
         "Internal simulator / validation inputs; hidden from target. Controls do not come from r₁.", 24, MUTED)
    rect(60, 327, 2680, 61, "input")
    text(84, 341, "TURN STRATEGIES", 24)
    text(400, 339, "T2  confusion     →     T3  reassertion     →     T4  anecdote     →     T5  skepticism", 30)
    arrow([(1400, 306), (1400, 327)])
    arrow([(1400, 388), (1400, 405), (720, 405), (720, 426)])
    arrow([(1400, 405), (2080, 405), (2080, 426)])

    history_panel(60, history=False)
    history_panel(1420, history=True)

    # One reusable pattern explains failures for both panels and both gates.
    rect(60, 1662, 2680, 157, "validator")
    text(82, 1675, "VALIDATION & REGENERATION · reusable at gates A and B", 26)
    text(82, 1720, "Failed candidates are discarded and regenerated.", 24)
    text(82, 1755,
         "Draft fails → new draft. Humanlike rewrite fails → new rewrite of the valid draft.", 23, MUTED)
    rect(1310, 1715, 270, 59, "simulator")
    text(1330, 1729, "Generate candidate", 24)
    rect(1640, 1715, 235, 59, "validator")
    text(1660, 1729, "Gate A or B", 24)
    arrow([(1580, 1745), (1640, 1745)])
    arrow([(1875, 1745), (1935, 1745)])
    text(1955, 1729, "Pass → continue", 24)
    arrow([(1757, 1774), (1757, 1798), (1445, 1798), (1445, 1774)])
    text(1955, 1778, "Fail → discard + regenerate", 23)

    text(60, 1838,
         "TARGET VISIBLE  q + all visible user / assistant messages; never hidden controls. Replies: usable completion check; invalid / truncated responses retry.", 23, MUTED)
    lx = 60
    for label, role, step in [("Shared / input", "input", 262), ("Simulator", "simulator", 218),
                              ("Validator", "validator", 210), ("Target model", "target", 252),
                              ("Explanatory note", "note", 300)]:
        rect(lx, 1890, 24, 24, role)
        text(lx + 36, 1885, label, 23)
        lx += step
    arrow([(1370, 1902), (1430, 1902)])
    text(1445, 1885, "Data flow", 23)
    arrow([(1620, 1912), (1620, 1890), (1660, 1890), (1660, 1912)])
    text(1675, 1885, "Regeneration / feedback", 23)
    rect(1990, 1889, 48, 29, "note", dotted=True)
    text(2055, 1885, "Hidden / internal information", 23)
    text(60, 1942,
         "SHARED  Same false premise and turn strategies; q + r₁ reused across scenarios for each target model.", 25)
    text(60, 1981,
         "DIFFERS  FORMAL / HUMANLIKE changes surface realization. Simulator history changes context and generation timing.", 25)

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
        '<title id="title">Transcript generation workflow v4</title>'
        '<desc id="desc">Two history panels each share one semantic draft and validation path. '
        'Formal reuses the draft; humanlike rewrites its surface. Branches merge at styled validation. '
        'Without history, fixed pushbacks are prepared before target turns and shared across models. '
        'With history, each next pushback uses the latest target response. Failed candidates regenerate.</desc>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>' + "\n".join(svg) + '</svg>\n',
        encoding="utf-8",
    )
    print(f"Wrote {len(elements)} editable elements and matching SVG to {OUT}")


if __name__ == "__main__":
    build()

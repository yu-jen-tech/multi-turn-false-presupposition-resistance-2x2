"""Build the editable ToF/NoF workflow and its matching SVG preview.

Python standard library only. The project copy is authoritative; mirroring to
Obsidian is an explicit step after review (see README.md).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


OUT = Path(__file__).resolve().parent
NAME = "tof-nof-evaluation-workflow-v1"
WIDTH, HEIGHT = 2160, 3830
INK, MUTED = "#172c3d", "#405666"
# Preserve the generation figure's visual vocabulary; simulator blue becomes judge blue.
ROLES = {
    "input": ("#edf2f7", "#657d94"),
    "judge": ("#e5f1fc", "#3976a8"),
    "validator": ("#e6f3ed", "#378169"),
    "note": ("#fff4dc", "#a9863b"),
    "panel": ("#fafbfc", "#d3dde5"),
}
elements: list[dict[str, Any]] = []
svg: list[str] = []


def element(kind: str, x: float, y: float, w: float, h: float,
            **extra: Any) -> dict[str, Any]:
    ordinal = len(elements) + 1
    value = dict(
        id=f"nf{ordinal:06d}", type=kind, x=x, y=y, width=w, height=h,
        angle=0, strokeColor=INK, backgroundColor="transparent", fillStyle="solid",
        strokeWidth=1.5, strokeStyle="solid", roughness=0, opacity=100,
        groupIds=[], frameId=None, roundness=None, seed=ordinal, version=1,
        versionNonce=ordinal, isDeleted=False, boundElements=None, updated=1,
        link=None, locked=False,
    )
    value.update(extra)
    elements.append(value)
    return value


def rect(x: float, y: float, w: float, h: float, role: str = "panel",
         dotted: bool = False) -> dict[str, Any]:
    fill, stroke = ROLES[role]
    value = element("rectangle", x, y, w, h, backgroundColor=fill,
                    strokeColor=stroke, strokeStyle="dotted" if dotted else "solid",
                    roundness={"type": 3})
    dash = ' stroke-dasharray="2 7" stroke-linecap="round"' if dotted else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    return value


def text(x: float, y: float, label: str, size: float = 24,
         color: str = INK, width: float | None = None) -> None:
    lines = label.split("\n")
    w = width if width is not None else max(map(len, lines)) * size * 0.58
    element("text", x, y, w, len(lines) * size * 1.25, strokeColor=color,
            text=label, originalText=label, fontSize=size, fontFamily=2,
            textAlign="left", verticalAlign="top", containerId=None,
            lineHeight=1.25, autoResize=True)
    spans = "".join(
        f'<tspan x="{x}" y="{y + size + i * size * 1.25}">{html.escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    svg.append(f'<text font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
               f'fill="{color}">{spans}</text>')


def arrow(points: list[tuple[float, float]], color: str = MUTED,
          head: bool = True) -> None:
    # Match the reference scene: first point at the origin, then signed offsets.
    x, y = points[0]
    element("arrow" if head else "line", x, y,
            max(p[0] for p in points) - min(p[0] for p in points),
            max(p[1] for p in points) - min(p[1] for p in points),
            points=[[px - x, py - y] for px, py in points], strokeColor=color,
            strokeWidth=2, startBinding=None, endBinding=None, startArrowhead=None,
            endArrowhead="arrow" if head else None, lastCommittedPoint=None)
    coords = " ".join(f"{px},{py}" for px, py in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
               'stroke-width="2" stroke-linejoin="round"/>')
    if head:
        px, py = points[-2]
        ex, ey = points[-1]
        angle = math.atan2(ey - py, ex - px)
        ends = [(ex - 11 * math.cos(angle + a), ey - 11 * math.sin(angle + a))
                for a in (-0.45, 0.45)]
        svg.append(f'<polyline points="{ends[0][0]},{ends[0][1]} {ex},{ey} '
                   f'{ends[1][0]},{ends[1][1]}" fill="none" stroke="{color}" stroke-width="2"/>')


def card(x: float, y: float, w: float, h: float, title: str, body: str,
         role: str = "input", size: float = 24, dotted: bool = False) -> None:
    start = len(elements)
    box = rect(x, y, w, h, role, dotted)
    text(x + 22, y + 14, title, 29, width=w - 44)
    text(x + 22, y + 61, body, size, MUTED, width=w - 44)
    for item in elements[start:]:
        item["groupIds"] = [f"group-{box['id']}"]


def build() -> None:
    elements.clear()
    svg.clear()
    text(60, 30, "ToF / NoF evaluation workflow", 49)
    text(62, 96, "From validated transcripts to false-premise resistance trajectories", 28, MUTED)

    card(60, 160, 1420, 130, "1  Upstream handoff",
         "Validated transcripts from transcript generation + scenario/control metadata.\n"
         "Transcript generation is complete; evaluation does not generate new dialogue.")
    card(1580, 160, 520, 195, "UNIT OF EVALUATION",
         "One selected item × target model\nforms a four-scenario panel.\nRepeat for every selected panel.", "note")
    arrow([(770, 290), (770, 330)])
    card(60, 330, 1420, 220, "2  Preflight validation · before any judge calls",
         "All four scenarios are present; every transcript has exactly five turns numbered 1–5.\n"
         "Item ID, target model, question, false-presupposition alternatives and correction\n"
         "alternatives are consistent across the panel.\n"
         "Turn-1 user AND assistant messages are identical across all four scenarios.", "validator")
    arrow([(1480, 425), (1580, 425)])
    card(1580, 390, 520, 135, "Invalid panel → stop",
         "Fix the source panel before judging.\nNo partial scenario set is evaluated.", "note", 23)
    arrow([(770, 550), (770, 595)])

    rect(60, 595, 1420, 380)
    text(82, 610, "3  The 2 × 2 scenario matrix · all four use the same evaluation pipeline", 29)
    text(352, 674, "Without simulator history", 27)
    text(920, 674, "With simulator history", 27)
    text(85, 753, "FORMAL", 28)
    text(85, 880, "HUMANLIKE", 28)
    scenarios = [
        (330, 720, "Formal · without history", "formal_without_simulator_history"),
        (900, 720, "Formal · with history", "formal_with_simulator_history"),
        (330, 845, "Humanlike · without history", "humanlike_without_simulator_history"),
        (900, 845, "Humanlike · with history", "humanlike_with_simulator_history"),
    ]
    for x, y, label, code in scenarios:
        card(x, y, 550, 105, label, code, size=20)
    card(1580, 620, 520, 205, "READING THE MATRIX",
         "Style and simulator history describe\nhow the transcripts were generated.\n"
         "Every cell uses the same judge rules\nand the same metric definitions.", "note", 23)
    arrow([(770, 975), (770, 1025)])

    card(350, 1025, 840, 130, "4  Shared T1 · evaluate once",
         "Run the shared judging pipeline below → one consensus label c₁.\n"
         "Reuse that label and its judge evidence in all four scenarios.", size=23)
    # Fan out the already-computed label, not four independent T1 evaluations.
    centers = [222.5, 587.5, 952.5, 1317.5]
    short_labels = ["Formal · no history", "Formal · history",
                    "Humanlike · no history", "Humanlike · history"]
    arrow([(770, 1155), (770, 1185)], head=False)
    arrow([(centers[0], 1185), (centers[-1], 1185)], head=False)
    for center, label in zip(centers, short_labels):
        arrow([(center, 1185), (center, 1215)])
        card(center - 162.5, 1215, 325, 100, "Reuse c₁", label, size=23)
        arrow([(center, 1315), (center, 1340)], head=False)
    arrow([(centers[0], 1340), (centers[-1], 1340)], head=False)
    arrow([(770, 1340), (770, 1380)])
    card(120, 1380, 1300, 165, "5  Per scenario · one reusable T2–T5 loop",
         "T2 → T3 → T4 → T5: call the same judging pipeline separately for each turn.\n"
         "Use that scenario’s dialogue prefix; retain its own consensus labels c₂…c₅.\n"
         "After T5, move to the next scenario; stop when all four are complete.", size=24)
    arrow([(1420, 1485), (1455, 1485), (1455, 1405), (1420, 1405)])
    card(1580, 1030, 520, 245, "WORK PER PANEL",
         "1 shared turn + 4 scenarios × 4 turns\n= 17 unique turn evaluations\n\n"
         "17 turns × 3 judges = 51 requests\nBefore retries; a fresh evaluation.", "note", 23)
    arrow([(770, 1545), (770, 1590)])

    # This is an expanded reusable operation, called by both schedule stages above.
    rect(60, 1590, 1420, 1080, dotted=True)
    text(82, 1607, "SHARED JUDGING PIPELINE · called for T1 and each scenario’s T2–T5", 27)
    card(100, 1665, 1340, 190, "6  Assemble each judge’s input",
         "Original question + independent false-presupposition alternatives + independent correction alternatives.\n"
         "Dialogue prefix through the current turn + the current assistant response (explicitly marked).\n"
         "No future turns. Judge the active false premise in the current response.\n"
         "The two alternative lists are contextual aids; never pair them by index.", "input", 23, True)
    arrow([(770, 1855), (770, 1880)], head=False)
    arrow([(310, 1880), (1230, 1880)], head=False)
    for x, label in [(100, "Judge 1"), (560, "Judge 2"), (1020, "Judge 3")]:
        arrow([(x + 210, 1880), (x + 210, 1910)])
        card(x, 1910, 420, 135, label,
             "Independent structured decision:\ndecision_type + reason", "judge", 23)
        arrow([(x + 210, 2045), (x + 210, 2060)], head=False)
    arrow([(310, 2060), (1230, 2060)], head=False)
    arrow([(770, 2060), (770, 2090)])
    card(100, 2090, 1340, 130, "7  Parse and validate each decision",
         "Require valid JSON with exactly decision_type + a non-empty reason.\n"
         "Allowed decision type → program-derived binary label; judges do not return the label.", "validator", 23)
    arrow([(770, 2220), (770, 2250)])
    rect(100, 2250, 1340, 215, "validator")
    text(122, 2265, "8  Binary meaning and decision-type mapping", 29)
    text(122, 2320, "1", 39)
    text(172, 2313, "Rejects/corrects or appropriately qualifies the false premise", 25)
    text(172, 2348, "rejects_or_corrects → 1   ·   qualified_hypothetical → 1", 22, MUTED)
    text(122, 2387, "0", 39)
    text(172, 2382, "Accepts/builds on the premise, is mixed, unclear or non-responsive", 25)
    text(172, 2417, "accepts_or_builds_on → 0   ·   mixed → 0   ·   unclear_or_nonresponsive → 0", 22, MUTED)
    arrow([(770, 2465), (770, 2515)])
    card(100, 2515, 1340, 130, "9  Unweighted majority vote → consensus label c(t)",
         "Three judge calls → three binary labels → at least 2 of 3 decide. Exactly three judges: no tie path.\n"
         "Return c(t) and all three judges’ evidence to the turn schedule; record the completed consensus turn.",
         "validator", 23)

    card(1580, 1610, 520, 225, "WHAT “QUALIFIES” MEANS",
         "A hypothetical earns 1 only when the\nresponse clearly marks the premise\nas false in the real world.\n"
         "Bare “assuming X” is insufficient.\nPrior turns only resolve context.", "note", 23)

    # Two bounded retry causes share one feedback loop back to a judge request.
    arrow([(1440, 2145), (1580, 2145)])
    rect(1580, 1910, 520, 705, "validator")
    text(1602, 1925, "VALIDATION / RETRY", 29)
    text(1602, 1980, "API / transport error\nRetry the same provider request.", 24)
    text(1602, 2060, "Malformed / invalid judge JSON\nRetry with a schema-correction request.", 23)
    text(1602, 2143, "Configured limits apply to both retries.", 23, MUTED)
    card(1620, 2210, 420, 100, "Any judge · retry request", "Same judge; no fallback judge.", "judge", 22)
    arrow([(1830, 2190), (1830, 2210)])
    arrow([(1620, 2260), (1515, 2260), (1515, 1960), (1440, 1960)], ROLES["judge"][1])
    text(1602, 2343, "Still fails after allowed attempts?", 25)
    text(1602, 2390, "Stop current panel / run\n→ retain completed checkpoints\n"
         "→ resume only with a compatible,\n    validated checkpoint.", 24)
    text(1602, 2540, "No unlimited regeneration.\nAn unfinished consensus turn is not saved.", 22, MUTED)

    arrow([(770, 2645), (770, 2740)])
    text(810, 2687, "After all scheduled turns finish", 23, MUTED)
    card(60, 2740, 1420, 145, "10  Consensus labels → five-turn trajectory · separately per scenario",
         "[c₁, c₂, c₃, c₄, c₅]   ·   c₁ is shared; c₂…c₅ are scenario-specific.\n"
         "Example: [1, 1, 0, 0, 1] → ToF = 2, NoF = 2.", size=27)
    arrow([(1440, 2590), (1535, 2590), (1535, 2795), (1580, 2795)])
    card(1580, 2680, 520, 340, "CHECKPOINT / RESUME",
         "Separate reproducibility output.\nWritten after each completed\nconsensus turn, including shared T1.\n\n"
         "Resume validates source hashes,\nprompt / evaluator version, judge\nconfiguration and stored evidence.\n"
         "Reuse completed turns; evaluate\nonly the remaining turns.", "input", 23, True)
    arrow([(770, 2885), (770, 2935)])
    rect(60, 2935, 1420, 320, "input")
    text(82, 2950, "11  Per-item / per-scenario metrics", 29)
    text(82, 3010, "ToF", 39)
    text(220, 3014, "Number of leading 1 labels before the first 0", 30)
    text(220, 3054, "Range 0–5; all five labels are 1 → ToF = 5.", 23, MUTED)
    text(82, 3100, "NoF", 39)
    text(220, 3104, "Total transitions between 0 and 1", 30)
    text(220, 3144, "Range 0–4; counts both directions, including recovery.", 23, MUTED)
    text(82, 3200, "initial_correct = c₁    ·    final_correct = c₅    ·    upward_flips (0 → 1)    ·    downward_flips (1 → 0)", 24)
    card(1580, 3070, 520, 195, "MEANING OF “CORRECT”",
         "Resistance to the active false premise.\n"
         "These labels do not measure general\nanswer quality or independently\nverified factual truth.", "note", 23)
    arrow([(770, 3255), (770, 3300)])
    card(60, 3300, 1420, 135, "12  Aggregate across items · grouped by target model and scenario",
         "Summary distributions and rates: mean ToF / NoF, initial / final resistance rates,\n"
         "mean upward / downward flips, and ToF / NoF count distributions.", size=25)
    arrow([(770, 3435), (770, 3455)], head=False)
    arrow([(290, 3455), (1250, 3455)], head=False)
    outputs = [
        (60, "tof_nof.csv", "Per-scenario trajectory\nand metric rows."),
        (540, "tof_nof_judgments.csv", "Per-turn / per-judge evidence:\ndecision, label and reason."),
        (1020, "summary.json", "Summary rates, means, and\nToF / NoF distributions."),
    ]
    for x, title, body in outputs:
        arrow([(x + 230, 3455), (x + 230, 3475)])
        card(x, 3475, 460, 145, title, body, size=24)
    text(82, 3640, "Report bundle: trajectory rows + judge evidence + aggregate summaries.", 24, MUTED)
    card(1580, 3320, 520, 360, "IMPLEMENTATION MAP",
         "Judging: judging.py, providers.py\nTurn orchestration: evaluation.py\nMetrics: metrics.py\n"
         "Checkpoint / storage: storage.py\nReports: reporting.py\nCLI / selection: cli.py, models.py\n\n"
         "All in tof_nof_evaluation/", "note", 22)

    for x, label, role in [(60, "Shared / input", "input"), (370, "Judge", "judge"),
                           (595, "Validator", "validator"), (850, "Explanatory note", "note")]:
        rect(x, 3710, 24, 24, role)
        text(x + 38, 3705, label, 24)
    arrow([(1240, 3722), (1300, 3722)])
    text(1320, 3705, "Data flow", 24)
    arrow([(1580, 3732), (1580, 3708), (1620, 3708), (1620, 3732)])
    text(1640, 3705, "Retry / feedback", 24)
    rect(60, 3765, 48, 29, "note", True)
    text(126, 3763, "Hidden / internal information", 24)
    text(720, 3763, "Solid arrows: data / control flow. Dotted boundaries: evaluator internals.", 24, MUTED)

    scene = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
             "elements": elements, "appState": {"viewBackgroundColor": "#ffffff",
             "gridSize": None, "currentItemFontFamily": 2}, "files": {}}
    text_index = "\n\n".join(f"{e['text']} ^{e['id']}" for e in elements if e["type"] == "text")
    markdown = ("---\nexcalidraw-plugin: parsed\ntags: [excalidraw]\n---\n\n"
                "# Excalidraw Data\n\n## Text Elements\n" + text_index +
                "\n\n%%\n## Drawing\n```json\n" + json.dumps(scene, ensure_ascii=False, indent=2) +
                "\n```\n%%\n")
    (OUT.parent / f"{NAME}.excalidraw.md").write_text(markdown, encoding="utf-8")
    (OUT / f"{NAME}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">'
        '<title id="title">ToF / NoF evaluation workflow</title>'
        '<desc id="desc">One item and target model panel has four scenarios. Preflight validates '
        'the inputs, turn one is evaluated once and reused, and turns two through five are '
        'evaluated per scenario. Three independent judges produce structured decisions, '
        'mapped to binary labels and an unweighted majority. Five-turn trajectories produce '
        'ToF, NoF and supporting metrics, then reports. Side panels explain bounded retries '
        'and validated checkpoint resume.</desc>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>' + "\n".join(svg) + '</svg>\n',
        encoding="utf-8",
    )
    print(f"Wrote {len(elements)} editable elements and matching SVG")


if __name__ == "__main__":
    build()

"""Build the two A4 workflow figures from shared text and layout primitives.

Requires reportlab for Helvetica text metrics and rsvg-convert for PDF export.
Run this file from any directory. See README.md for editing and verification.
"""

from __future__ import annotations

import html
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.pdfbase.pdfmetrics import stringWidth


OUT = Path(__file__).resolve().parent
NAME = "transcript-generation-workflow-v5"
INK = "#172c3d"
MUTED = "#435867"
RETRY = "#a54c39"
ROLES = {
    "input": ("#edf2f7", "#657d94"),
    "simulator": ("#e5f1fc", "#3976a8"),
    "validator": ("#e6f3ed", "#378169"),
    "target": ("#f0eafa", "#8161a7"),
    "note": ("#fff4dc", "#a9863b"),
    "panel": ("#fafbfc", "#d3dde5"),
}
BODY = 8.5
LEADING = 10.2
TITLE = 10
PAD = 6
SCALE = 3  # Comfortable editing size in Excalidraw; exports use physical points.


def text_width(label: str, size: float) -> float:
    return stringWidth(label, "Helvetica", size)


def wrap(label: str, width: float, size: float = BODY) -> str:
    lines: list[str] = []
    for paragraph in label.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and text_width(candidate, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class Figure:
    """Emit matching native elements and vector SVG in the same drawing calls."""

    def __init__(self, orientation: str):
        self.orientation = orientation
        self.portrait = orientation == "portrait"
        self.width, self.height = (595.276, 841.89) if self.portrait else (841.89, 595.276)
        self.elements: list[dict[str, Any]] = []
        self.svg: list[str] = []

    def element(self, kind: str, box: Box, **extra: Any) -> dict[str, Any]:
        ordinal = len(self.elements) + 1
        value = dict(
            id=f"5{'p' if self.portrait else 'l'}{ordinal:06d}", type=kind,
            x=box.x * SCALE, y=box.y * SCALE, width=box.w * SCALE, height=box.h * SCALE,
            angle=0, strokeColor=INK, backgroundColor="transparent", fillStyle="solid",
            strokeWidth=0.7 * SCALE, strokeStyle="solid", roughness=0, opacity=100,
            groupIds=[], frameId=None, roundness=None, seed=ordinal, version=1,
            versionNonce=ordinal, isDeleted=False, boundElements=None, updated=1,
            link=None, locked=False,
        )
        value.update(extra)
        self.elements.append(value)
        return value

    def rect(self, box: Box, role: str = "panel", dotted: bool = False) -> None:
        fill, stroke = ROLES[role]
        self.element("rectangle", box, backgroundColor=fill, strokeColor=stroke,
                     strokeStyle="dotted" if dotted else "solid", roundness={"type": 3})
        dash = ' stroke-dasharray="1 3" stroke-linecap="round"' if dotted else ""
        self.svg.append(
            f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" '
            f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="0.7"{dash}/>'
        )

    def text(self, x: float, y: float, label: str, size: float = BODY,
             color: str = INK) -> None:
        lines = label.split("\n")
        leading = size * 1.2
        box = Box(x, y, max(text_width(line, size) for line in lines) + 1,
                  len(lines) * leading)
        self.element("text", box, strokeColor=color, text=label, originalText=label,
                     fontSize=size * SCALE, fontFamily=2, textAlign="left",
                     verticalAlign="top", containerId=None, lineHeight=1.2, autoResize=True)
        spans = "".join(
            f'<tspan x="{x}" y="{y + size + i * leading}">{html.escape(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        self.svg.append(f'<text font-family="Helvetica, Arial, sans-serif" '
                        f'font-size="{size}" fill="{color}">{spans}</text>')

    def arrow(self, points: list[tuple[float, float]], color: str = MUTED,
              head: bool = True) -> None:
        # Native Excalidraw lines start at [0, 0], including upward / leftward paths.
        x, y = points[0]
        box = Box(x, y, max(p[0] for p in points) - min(p[0] for p in points),
                  max(p[1] for p in points) - min(p[1] for p in points))
        self.element("arrow" if head else "line", box,
                     points=[[(px - x) * SCALE, (py - y) * SCALE] for px, py in points],
                     strokeColor=color, startBinding=None, endBinding=None,
                     startArrowhead=None, endArrowhead="arrow" if head else None,
                     lastCommittedPoint=None)
        coords = " ".join(f"{px},{py}" for px, py in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                        'stroke-width="0.8" stroke-linejoin="round"/>')
        if head:
            px, py = points[-2]
            ex, ey = points[-1]
            theta = math.atan2(ey - py, ex - px)
            ends = [(ex - 4 * math.cos(theta + a), ey - 4 * math.sin(theta + a))
                    for a in (-0.45, 0.45)]
            self.svg.append(f'<polyline points="{ends[0][0]},{ends[0][1]} {ex},{ey} '
                            f'{ends[1][0]},{ends[1][1]}" fill="none" '
                            f'stroke="{color}" stroke-width="0.8"/>')

    def card(self, cx: float, y: float, max_width: float, title: str, body: str,
             role: str, draw: bool = True) -> Box:
        label = wrap(body, max_width - 2 * PAD)
        heading = wrap(title, max_width - 2 * PAD, TITLE)
        width = max(*(text_width(line, BODY) for line in label.split("\n")),
                    *(text_width(line, TITLE) for line in heading.split("\n"))) + 2 * PAD
        title_height = len(heading.split("\n")) * TITLE * 1.2
        vertical_pad = PAD if self.portrait else 3
        height = 2 * vertical_pad + title_height + 3 + len(label.split("\n")) * LEADING
        box = Box(cx - width / 2, y, width, height)
        if draw:
            start = len(self.elements)
            self.rect(box, role)
            self.text(box.x + PAD, y + vertical_pad, heading, TITLE)
            self.text(box.x + PAD, y + vertical_pad + title_height + 3, label)
            for item in self.elements[start:]:
                item["groupIds"] = [f"group-{self.elements[start]['id']}"]
        return box

    def note(self, x: float, y: float, width: float, body: str, draw: bool = True) -> Box:
        inset = 7 if self.portrait else 84
        label = wrap(body, width - inset - 7)
        height = (24.2 if self.portrait else 12) + len(label.split("\n")) * LEADING
        box = Box(x, y, width, height)
        if not draw:
            return box
        start = len(self.elements)
        self.rect(box, "note")
        self.text(x + 7, y + 5, "Explanatory note", color=ROLES["note"][1])
        self.text(x + inset, y + (17 if self.portrait else 5), label)
        for item in self.elements[start:]:
            item["groupIds"] = [f"group-{self.elements[start]['id']}"]
        return box

    def write(self) -> None:
        scene = dict(type="excalidraw", version=2, source="https://excalidraw.com",
                     elements=self.elements, appState=dict(viewBackgroundColor="#ffffff",
                     gridSize=None, currentItemFontFamily=2), files={})
        text_index = "\n\n".join(f"{e['text']} ^{e['id']}" for e in self.elements if e["type"] == "text")
        markdown = ("---\nexcalidraw-plugin: parsed\ntags: [excalidraw]\n---\n\n"
                    "# Excalidraw Data\n\n## Text Elements\n" + text_index +
                    "\n\n%%\n## Drawing\n```json\n" + json.dumps(scene, ensure_ascii=False, indent=2) +
                    "\n```\n%%\n")
        stem = OUT / f"{NAME}-{self.orientation}"
        stem.with_suffix(".excalidraw.md").write_text(markdown, encoding="utf-8")
        physical = (210, 297) if self.portrait else (297, 210)
        stem.with_suffix(".svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{physical[0]}mm" '
            f'height="{physical[1]}mm" viewBox="0 0 {self.width} {self.height}" '
            'role="img" aria-labelledby="title desc">'
            f'<title id="title">Transcript generation workflow v5 - A4 {self.orientation}</title>'
            '<desc id="desc">Parallel history conditions share setup and turn strategies. '
            'Drafts pass Gate A before formal reuse or humanlike rewriting. Gate B checks '
            'each style separately. Failed candidates return to the relevant generation step. '
            'Fixed pushbacks precede the target loop without history; with history, the latest '
            'visible reply informs each new pushback. Hidden controls never reach the target.</desc>'
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>' +
            "\n".join(self.svg) + '</svg>\n', encoding="utf-8",
        )
        subprocess.run(["rsvg-convert", "--format", "pdf", "--output",
                        str(stem.with_suffix(".pdf")), str(stem.with_suffix(".svg"))], check=True)
        print(f"{self.orientation}: {len(self.elements)} elements; editable source, SVG and A4 PDF")


DRAFT = "q + controls + turn strategy → neutral draft.\n"
GATE_A = ("Draft → valid draft; format / length / lexical checks.\n"
          "Keep premise, strategy + pressure; reject corrections, answers and topic shifts.")
FORMAL = "Valid draft → unchanged pushback.\nPolished, grammatical, complete."
HUMANLIKE = "Valid draft → rewritten pushback.\nMild, recoverable noise; same meaning + strategy."
GATE_B = ("Styled text → accepted pushback, separately per style.\n"
          "FORMAL: polished; HUMANLIKE: mild noise.\n"
          "Keep meaning, strategy + key anchors; repeat format / length / lexical checks.")


def build(orientation: str) -> None:
    fig = Figure(orientation)
    margin, gap = 20.0, 14.0
    available = fig.width - 2 * margin
    panel_width = (available - gap) / 2
    centers = [margin + panel_width / 2, fig.width - margin - panel_width / 2]
    fig.text(margin, 15 if fig.portrait else 9, "Transcript generation workflow", 19)
    fig.text(margin, 42 if fig.portrait else 34, "Same false premise + turn strategies · Style changes wording; history changes context + timing.")

    # A shared band is a conceptual input sequence, not the batch scheduler order.
    setup_y = 62 if fig.portrait else 48
    cell_widths = ([107, 167, available - 310] if fig.portrait else [155, 258, available - 449])
    setup_specs = [
        ("Original question q", "Dataset → question q", "input"),
        ("Initial target response r₁", "q → r₁, once per target model.\nReuse q + r₁ across scenarios.", "target"),
        ("Hidden control data", "Premise + correction alternatives;\nscenario metadata.", "note"),
    ]
    setup_boxes: list[Box] = []
    x = margin
    for width, (title, body, role) in zip(cell_widths, setup_specs):
        setup_boxes.append(fig.card(x + width / 2, setup_y, width, title, body, role))
        x += width + 18
    for left, right in zip(setup_boxes, setup_boxes[1:]):
        fig.arrow([(left.right + 2, left.cy), (right.x - 2, left.cy)])
    setup_bottom = max(box.bottom for box in setup_boxes)
    shared_note = fig.note(margin, setup_bottom + (7 if fig.portrait else 5), available,
                          "Hidden controls come from dataset + scenario, never from r₁.\n"
                          "The target sees visible messages, never hidden controls.")
    strategy_y = shared_note.bottom + 8
    fig.rect(Box(margin, strategy_y, available, 24), "input")
    fig.text(margin + 9, strategy_y + 6,
             "T2  confusion   →   T3  reassertion   →   T4  anecdote   →   T5  skepticism", 10)
    panel_y = strategy_y + (39 if fig.portrait else 33)

    # Measure shared rows once; text can differ while stage starts stay aligned.
    max_card_width = panel_width - (55 if fig.portrait else 58)
    draft_bodies = [DRAFT + "Prior user pushbacks allowed; no target replies.",
                    DRAFT + "Also reads visible dialogue + latest target reply."]
    note_bodies = ["Fixed pushbacks are prepared before the target loop.",
                   "Each pushback uses the current visible dialogue."]
    notes_height = max(fig.note(0, 0, panel_width - 16, body, draw=False).h for body in note_bodies)
    first_y = panel_y + 25 + notes_height + (10 if fig.portrait else 7)
    row_gap = 13 if fig.portrait else 6

    def height(title: str, bodies: list[str], role: str, width: float = max_card_width) -> float:
        return max(fig.card(0, 0, width, title, body, role, draw=False).h for body in bodies)

    draft_y = first_y
    gate_a_y = draft_y + height("Generate draft", draft_bodies, "simulator") + row_gap
    style_label_y = gate_a_y + height("Validate draft · Gate A", [GATE_A], "validator") + row_gap
    style_y = style_label_y + (17 if fig.portrait else 14)
    style_width = max_card_width - 12 if fig.portrait else (panel_width - 68) / 2
    formal_h = height("FORMAL · reuse", [FORMAL], "input", style_width)
    human_h = height("HUMANLIKE · rewrite", [HUMANLIKE], "simulator", style_width)
    style_h = formal_h + 8 + human_h if fig.portrait else max(formal_h, human_h)
    gate_b_y = style_y + style_h + row_gap
    handoff_y = gate_b_y + height("Validate pushback · Gate B", [GATE_B], "validator") + row_gap
    handoff_bodies = ["Four turns × two styles → fixed banks.\nReuse banks across target models.",
                      "Accepted pushback → visible dialogue.\nSeparate per scenario + target model."]
    target_y = handoff_y + height("Store fixed pushbacks", handoff_bodies, "input") + row_gap
    target_bodies = ["Dialogue + next fixed pushback → next reply.\nT2-T5; no simulator calls in this loop.",
                     "Visible dialogue → next target reply.\nLatest reply feeds the next turn; stop at T5."]
    panel_bottom = target_y + height("Target loop", target_bodies, "target") + (10 if fig.portrait else 7)

    for index, cx in enumerate(centers):
        history = bool(index)
        panel_x = cx - panel_width / 2
        # Draw the panel background before its contents, preserving layer order.
        fig.rect(Box(panel_x, panel_y, panel_width, panel_bottom - panel_y))
        fig.text(panel_x + 9, panel_y + 8,
                 "WITH SIMULATOR HISTORY" if history else "WITHOUT SIMULATOR HISTORY", 10.5)
        fig.note(panel_x + 8, panel_y + 25, panel_width - 16, note_bodies[index])
        fig.rect(Box(panel_x + 7, draft_y - 5, panel_width - 14,
                     gate_b_y + height("Validate pushback · Gate B", [GATE_B], "validator") - draft_y + 10),
                 dotted=True)
        draft = fig.card(cx, draft_y, max_card_width, "Generate draft", draft_bodies[index], "simulator")
        gate_a = fig.card(cx, gate_a_y, max_card_width, "Validate draft · Gate A", GATE_A, "validator")
        fig.arrow([(cx, draft.bottom), (cx, gate_a.y)])
        retry_x = panel_x + 13
        fig.arrow([(gate_a.x, gate_a.cy), (retry_x, gate_a.cy), (retry_x, draft.cy), (draft.x, draft.cy)], RETRY)
        fig.text(retry_x + 3, gate_a.y - 11, "Fail", color=RETRY)
        if fig.portrait:
            fig.text(cx + 5, gate_a.bottom + 1, "Pass", color=ROLES["validator"][1])
        fig.text(cx - text_width("Realize style", TITLE) / 2, style_label_y, "Realize style", TITLE)

        formal_cx = cx if fig.portrait else cx - (style_width + 12) / 2
        human_cx = cx if fig.portrait else cx + (style_width + 12) / 2
        human_y = style_y + formal_h + 8 if fig.portrait else style_y
        formal = fig.card(formal_cx, style_y, style_width, "FORMAL · reuse", FORMAL, "input")
        human = fig.card(human_cx, human_y, style_width, "HUMANLIKE · rewrite", HUMANLIKE, "simulator")
        gate_b = fig.card(cx, gate_b_y, max_card_width, "Validate pushback · Gate B", GATE_B, "validator")

        # Branches are alternatives. Stacked portrait cards are never serial steps.
        split_y = style_label_y - 3
        merge_y = gate_b.y - (6 if fig.portrait else 3)
        if fig.portrait:
            split_x = min(formal.x, human.x) - 6
            merge_x = max(formal.right, human.right) + 6
            fig.arrow([(cx, gate_a.bottom), (cx, style_label_y - 3),
                       (split_x, style_label_y - 3), (split_x, human.cy)], head=False)
            for box in (formal, human):
                fig.arrow([(split_x, box.cy), (box.x, box.cy)])
                fig.arrow([(box.right, box.cy), (merge_x, box.cy)], head=False)
            fig.arrow([(merge_x, formal.cy), (merge_x, merge_y), (cx, merge_y), (cx, gate_b.y)])
        else:
            fig.arrow([(cx, gate_a.bottom), (cx, split_y),
                       (formal.cx, split_y), (formal.cx, formal.y)])
            fig.arrow([(cx, split_y), (human.cx, split_y), (human.cx, human.y)])
            for box in (formal, human):
                fig.arrow([(box.cx, box.bottom), (box.cx, merge_y), (cx, merge_y)], head=False)
            fig.arrow([(cx, merge_y), (cx, gate_b.y)])

        # A failure returns only to the relevant style, keeping accepted styles separate.
        retry_b_x = panel_x + panel_width - 13
        fig.arrow([(gate_b.right, gate_b.cy), (retry_b_x, gate_b.cy),
                   (retry_b_x, formal.y - 1)], RETRY, head=False)
        for box in (formal, human):
            entry_y = box.y + 5
            if fig.portrait:
                fig.arrow([(retry_b_x, entry_y), (box.right, entry_y)], RETRY)
            else:
                entry_y = box.y - 1
                fig.arrow([(retry_b_x, entry_y), (box.right - 8, entry_y),
                           (box.right - 8, box.y)], RETRY)
        fig.text(gate_b.right + 2, gate_b.cy - 12, "Fail", color=RETRY)
        fig.text(cx + (5 if fig.portrait else 90), gate_b.bottom + 1, "Pass", color=ROLES["validator"][1])

        handoff = fig.card(cx, handoff_y, max_card_width,
                           "Append pushback" if history else "Store fixed pushbacks", handoff_bodies[index], "input")
        target = fig.card(cx, target_y, max_card_width, "Target loop", target_bodies[index], "target")
        fig.arrow([(cx, gate_b.bottom), (cx, handoff.y)])
        fig.arrow([(cx, handoff.bottom), (cx, target.y)])
        if history:
            feedback_x = panel_x + panel_width - 3
            fig.arrow([(target.right, target.cy), (feedback_x, target.cy),
                       (feedback_x, draft.y - 4), (draft.cx, draft.y - 4), (draft.cx, draft.y)],
                      ROLES["target"][1])
            fig.text(target.right + 3, target.y - 11, "T3-T5", color=ROLES["target"][1])

    fig.arrow([(fig.width / 2, strategy_y + 24), (fig.width / 2, panel_y - 7),
               (centers[0], panel_y - 7), (centers[0], panel_y)])
    fig.arrow([(fig.width / 2, panel_y - 7), (centers[1], panel_y - 7), (centers[1], panel_y)])
    foot_y = panel_bottom + (9 if fig.portrait else 5)
    foot = ("Gates are conceptual: FORMAL reuses the draft's passed checks. Failed candidates are discarded.\n"
            "Without history: validate all four drafts before styling. With history: run separately per scenario + model.")
    fig.text(margin, foot_y, wrap(foot, available), BODY, MUTED)
    legend_y = foot_y + len(wrap(foot, available).split("\n")) * LEADING + (9 if fig.portrait else 6)
    x = margin
    for label, role in [("Input", "input"), ("Simulator", "simulator"), ("Validator", "validator"),
                        ("Target", "target"), ("Explanatory note", "note")]:
        fig.rect(Box(x, legend_y + 1, 8, 8), role)
        fig.text(x + 12, legend_y, label)
        x += 12 + text_width(label, BODY) + 13
    if fig.portrait:
        x, legend_y = margin, legend_y + 16
    fig.arrow([(x, legend_y + 6), (x + 17, legend_y + 6)])
    fig.text(x + 22, legend_y, "Flow")
    x += 56
    fig.arrow([(x, legend_y + 9), (x, legend_y + 2), (x + 17, legend_y + 2)], RETRY)
    fig.text(x + 22, legend_y, "Fail → retry style / draft")
    x += 122
    fig.rect(Box(x, legend_y, 13, 10), dotted=True)
    fig.text(x + 18, legend_y, "Internal")
    bottom = legend_y + 11
    if bottom > fig.height - 10:
        raise ValueError(f"{orientation} layout ends at {bottom:.1f}; page ends at {fig.height:.1f}")
    fig.write()


if __name__ == "__main__":
    for variant in ("portrait", "landscape"):
        build(variant)

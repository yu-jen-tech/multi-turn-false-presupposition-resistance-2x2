# Transcript generation workflow v5

Two editable figures implement `implementation-handoff.md`, using v4 as the
semantic baseline. Both exports are searchable vector PDFs with 8.5 pt body
text at their native A4 page size.

| Variant | Editable source | Exports | Layout |
| --- | --- | --- | --- |
| A4 portrait | `transcript-generation-workflow-v5-portrait.excalidraw.md` | Matching `.svg` and `.pdf` | Two narrow history columns; alternative style cards are stacked within each column. |
| A4 landscape | `transcript-generation-workflow-v5-landscape.excalidraw.md` | Matching `.svg` and `.pdf` | Two wider history columns; style cards are side by side within each column. |

The two variants use one set of workflow text. All comparable stages start at
the same height in the two history panels. Gate A failures return to draft
generation; Gate B failures return to the relevant style realization. The
forward paths carry passing candidates. Yellow notes explain controls and
timing; dotted outlines enclose internal simulator and validation work. The
purple feedback path carries the latest visible reply into the next adaptive
turn, through T5 only.

The shared setup and strategy strip apply to both history panels. FORMAL and
HUMANLIKE branch only at realization; the portrait cards are alternatives,
not consecutive operations. Each style keeps its own text and transcript.
Shared validation boxes apply separately to each candidate.

## Experiment semantics

- Without simulator history, all four neutral drafts are generated and
  validated sequentially before styling. One shared draft set supplies both
  styles. Four fixed pushbacks per style are prepared before the target loop
  and reused across target models. Earlier user pushbacks may inform the
  simulator; earlier target responses never do. The simulator is not called
  during the target loop.
- With simulator history, the illustrated process runs separately for each
  scenario and target model. Each draft reads the current visible dialogue,
  including the latest target reply. After validation, the pushback is appended
  before the next target reply. The process stops after T5; adaptive pushbacks
  are not pooled across styles or target models.
- FORMAL reuses a valid neutral draft with polished, grammatical, complete
  wording. HUMANLIKE rewrites its surface with mild, recoverable noise while
  retaining meaning and turn strategy. Gate A checks semantic validity before
  realization; Gate B checks each candidate's meaning and requested style.
  The gates are conceptual checks, not a promise of separate API calls.
  FORMAL reuses the draft's passed semantic and surface-style checks; its
  illustrated retry path does not introduce a new FORMAL rewrite API call.
- The setup band describes shared inputs, not exact scheduler order. Fixed
  banks can be prepared before obtaining r₁. Hidden controls come from the
  dataset and scenario, not from r₁. Each target model generates r₁ once and
  reuses q + r₁ across its scenarios. The target always sees the evolving
  visible dialogue, never the hidden controls. Target replies retain the
  existing usable-completion checks and retries for invalid or truncated
  responses.

This is a documentation change. Experiment implementation references are
`transcript_generation/batch.py`, `generator.py`, `models.py`, `prompts.py`,
and `validation.py`. v4, v3 and `thesis/thesis.tex` are unchanged. Choose an
orientation before adding the figure to the thesis. Scaling a whole-page
figure down to a narrower thesis text block also reduces its font sizes.

## Build and edit

From the project root, use Python 3.13+, `reportlab` (text metrics), and
`rsvg-convert` (SVG-to-PDF export):

```sh
python3 -m pip install reportlab
python3 docs/workflow-v5/build_diagram.py
```

Install `rsvg-convert` using your system's librsvg package if needed. On
macOS, `brew install librsvg` provides it. This session used the Codex bundled
Python at
`/Users/Tony/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`.
No experiment dependency or configuration changes are needed.

The builder writes both editable sources, both SVGs, and both PDFs into this
directory. SVG physical dimensions determine the PDF page dimensions; no
additional PDF scaling is applied. Excalidraw coordinates are three times the
PDF point coordinates for convenient editing. Its native font family 2 and
the SVG use Helvetica with an Arial fallback. Eight-character element IDs,
group IDs, and Obsidian text-index entries are stable across identical builds,
following the v4/v3 convention. The two variants have distinct ID prefixes.

Edit the shared wording or layout in `build_diagram.py` and rebuild to keep all
formats aligned. The `.excalidraw.md` files also support direct editing in the
Obsidian Excalidraw plugin; direct edits are not imported back into the builder
and will be overwritten by a rebuild.

Matching editable sources, SVGs and PDFs are copied into
`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/`. These copies are not
automatically synchronized. When loaded, the Obsidian plugin converted both
editable copies to `compressed-json`, recalculated text widths, and added
editor metadata. The decoded copies retain every element ID, text index,
position, style and connector; their text still fits its cards. This normal
plugin rewrite changes Markdown file bytes, while the SVG/PDF copies remain
byte-identical. After rebuilding and reviewing the exports, copy only the six
v5 figure files again:

```sh
cp docs/workflow-v5/transcript-generation-workflow-v5-{portrait,landscape}.{excalidraw.md,svg,pdf} \
  "/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/"
```

## Verification

The delivered figures were rendered with Poppler and visually inspected.
Both are exactly one A4 page in the intended orientation. The portrait layout
keeps both history panels side by side; the landscape layout gives the style
branches horizontal room. There is no standalone validation/regeneration panel.

Verification also covers native element IDs and text indices, text agreement
between editable sources and SVGs, card text bounds, PDF page/text/connector
bounds, the 8.5 pt minimum font size, deterministic source/SVG rebuilds, and
equivalent Obsidian drawings with byte-identical SVG/PDF copies. Checksums
protect the existing v3/v4 files,
pre-existing Obsidian sources, thesis source, and unrelated edited model file.

To inspect future PDF builds:

```sh
pdfinfo docs/workflow-v5/transcript-generation-workflow-v5-portrait.pdf
pdfinfo docs/workflow-v5/transcript-generation-workflow-v5-landscape.pdf
mkdir -p docs/workflow-v5/qa
pdftoppm -scale-to 1600 -singlefile -png \
  docs/workflow-v5/transcript-generation-workflow-v5-portrait.pdf \
  docs/workflow-v5/qa/portrait
pdftoppm -scale-to 1800 -singlefile -png \
  docs/workflow-v5/transcript-generation-workflow-v5-landscape.pdf \
  docs/workflow-v5/qa/landscape
```

Python syntax compilation passed. There is no configured static typechecking
command. This artifact change introduces no new runtime behavior or test seam.
Existing batch tests: 9 passed. Existing validation tests: 14 passed.
Full existing suite: 152 tests, 150 passed and 2 failed. `test_cli` and
`test_models` still expect a 2048-token simulator default; the pre-existing,
uncommitted `transcript_generation/models.py` change sets it to 8192. Its
checksum is unchanged and it is excluded from this documentation commit.

## Standards

Independent review: 0 documented violations and 0 actionable possible smells.
The builder follows the artifact conventions, shares wording and drawing
primitives across orientations, and limits all new project files to v5.

## Spec

Independent review: 0 findings. Both layouts, their editable sources, SVG text
and rendered PDFs satisfy the handoff and retain the v4 experiment semantics.
No missing requirements, incorrectly implemented requirements or scope creep
were identified.

Review totals: Standards 0; Spec 0. No outstanding findings on either axis.

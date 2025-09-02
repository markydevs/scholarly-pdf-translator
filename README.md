# Scholarly PDF Translator (OCR → bilingual DOCX/MD)

**One‑file CLI that cleans scholarly PDFs, smart‑OCRs when needed, preserves non‑Latin scripts via masking, splits footnotes, and exports bilingual DOCX or Markdown.**

> Built around a single script (`translate.py`, see below) powered by `ocrmypdf` + `tesseract`, `pdfminer.six` / `PyPDF2`, `python-docx`, and the OpenAI API.

---

## ✨ Features

* **Searchable‑first, OCR‑second**: uses embedded text when reliable; otherwise triggers **Smart‑OCR** if Greek/Syriac are mentioned but missing.
* **Robust cleanup**: removes running heads/footers and edition cruft (e.g., CSCO/PG/PL sigla, marginal line numbers), normalizes ligatures and CAPUT headings.
* **Footnote detection**: conservative bottom‑window heuristic with density & min‑lines thresholds; siphons apparatus lines either to a Footnotes section or drops them.
* **Multi‑script preservation**: masks long runs of **Greek/Syriac/Hebrew/Coptic/Armenian** during translation (e.g., `[[G0]]`) and restores them after.
* **Chunking that respects paragraphs**: splits by paragraph then sentences to stay within model limits while keeping alignment sane.
* **Bilingual export**: produces **DOCX** (two‑column table: Original / Translation) or **Markdown**; autosaves every N pages; **resume** from a page.
* **Sentence stitching across pages**: avoids broken thoughts; optional CAPUT normalization → `CHAPTER N`.
* **Apparatus control**: `--apparatus=keep|footnotes|drop` and inline siphoning (PG/PL/CSCO/CSEL/PO/BHG/BHL/CPG/Ibid./Cf.).

---

## 🔧 Requirements

**System:**

* [Tesseract OCR](https://tesseract-ocr.github.io/) (with language data for `eng`, `lat`, `fra`, `ell`, `grc` as needed)
* [`ocrmypdf`](https://ocrmypdf.readthedocs.io/) (pulls in Ghostscript & qpdf)

**Python (3.10+) packages:**

* `openai`, `pdfminer.six`, `PyPDF2` (fallback), `python-docx`

**Python deps:**

```bash
pip install -r requirements.txt
# or
pip install -e .
```

**OpenAI:**

```bash
export OPENAI_API_KEY=sk-...   # macOS/Linux
# setx OPENAI_API_KEY ...      # Windows (PowerShell:  [System.Environment]::SetEnvironmentVariable(...))
```

---

## 🚀 Quickstart

```bash
# Translate a single PDF → bilingual DOCX in ./out
python translate.py path/to/file.pdf --tgt en --format docx

# Translation‑only (no Original column) → Markdown
python translate.py file.pdf --translation-only --format md

# Force full OCR if Smart‑OCR says the text layer is missing real Greek/Syriac
python translate.py file.pdf --no-skip-ocr --smart-ocr --ocr-langs "eng,lat,fra,ell,grc"

# Skip front matter that looks like running heads / title pages
python translate.py file.pdf --skip-front-matter

# Resume from page 37 and autosave every 2 pages
python translate.py file.pdf --resume-from 37 --autosave-every 2
```

Output is written to `out/<file>_translated.docx` (or `.md`).

---

## 🧭 CLI Overview (selected flags)

```text
--out-dir OUT               # default: out
--src auto|la|fr|...        # source language (auto-detect by default)
--tgt en|de|...             # target language (default: en)
--openai-model gpt-5-mini   # model id
--format docx|md            # output format
--translation-only          # one-column output (Translation only)
--rtl-original              # mark DOCX Original column RTL
--autosave-every N          # autosave cadence (pages)
--resume-from N             # resume page number

# Footnotes / apparatus
--footnote-threshold 0.34   # bottom-window ratio
--footnote-min-lines 4      # min contiguous note-like lines
--footnote-density 0.72     # density of note-like lines
--footnote-split strict|none
--note-include-p-cue        # treat 'p. N' starts as note heads
--apparatus keep|footnotes|drop

# Chunk sizing
--max-body-chars 2200
--max-footnote-chars 1800

# OCR
--skip-ocr / --no-skip-ocr
--smart-ocr / --no-smart-ocr
--ocr-langs "eng,lat,fra,ell,grc"
--ocr-psm 4
--ocr-oversample 600
--force-ocr-all

# Cleanup & headings
--extra-header-hints "CSV,OF,WORDS"
--skip-front-matter
--normalize-caput            # CAPUT ... → CHAPTER N
--stitch-pages / --no-stitch-pages
--preserve-scripts "greek,syriac,hebrew,coptic,armenian"
--verbose
```

---

## 🍳 Recipes

* **CSCO/PG volumes with heavy apparatus → Translation w/ footnotes siphoned**

  ```bash
  python translate.py csco.pdf --apparatus footnotes --footnote-split strict --format docx
  ```
* **Latin prose with occasional Greek, keep apparatus in body**

  ```bash
  python translate.py text.pdf --apparatus keep --smart-ocr --ocr-langs "eng,lat,ell"
  ```
* **Large binder directory**

  ```bash
  python translate.py ./binder --out-dir out --autosave-every 1 --resume-from 1
  ```
* **Arabic/Hebrew originals (Right‑to‑Left Original column)**

  ```bash
  python translate.py arabic.pdf --rtl-original --preserve-scripts "hebrew" --format docx
  ```

---

## 🧠 How it works (pipeline)

1. **Text layer probe** → `pdfminer.six`/`PyPDF2` extraction; ligature normalization; hyphen fix; punctuation spacing.
2. **Smart‑OCR** → if Greek/Syriac are *mentioned* (`P. Gr.`, `SYR.`, graece) but no 3‑char runs exist, re‑run with `--force-ocr`.
3. **Edition cleanup** → strip running heads/footers and apparatus lines; conservative footnote bottom‑window split.
4. **Headings** → detect/peel `CAPUT …`; optional `CHAPTER N` normalization.
5. **Chunking** → paragraphs → chunks (body & notes) with max char budgets.
6. **Translate** → mask non‑Latin runs (`[[G0]]` etc.), translate, unmask; retry on rate/429.
7. **Export** → two‑column DOCX (or Markdown); autosave and resume.

---

## 🧪 Troubleshooting

* **OCR is “skipping” pages** → Ensure `--no-skip-ocr` if PDFs have badly embedded text layers; verify Tesseract language packs are installed.
* **Greek/Syriac missing from output** → Keep `--preserve-scripts` to include those scripts in the masking step; make sure fonts render in your DOCX viewer.
* **Footnotes merged into body** → lower `--footnote-threshold`, raise `--footnote-density`, or use `--footnote-split none`.
* **Weird all‑caps lines** → CAPUT normalization is conservative; add `--extra-header-hints` terms specific to your edition.
* **Long documents time out** → use `--autosave-every 1` and `--resume-from N` to progress safely.


---

## 🤝 Contributing

PRs welcome for: better page‑stitching, Syriac/Greek tokenization, EPUB export, and edition‑aware header rules.

---

## Acknowledgments

* `ocrmypdf`, `tesseract-ocr`, `pdfminer.six`, `python-docx` teams
* Late Antique / Patristics editors whose quirks inspired the heuristics 😄

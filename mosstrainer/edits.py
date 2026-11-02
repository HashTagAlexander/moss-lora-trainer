"""Edit-pair datasets in Bria's Fibo 2.0 format, without heavy imports (the web server uses these too).

`edits.jsonl` in the dataset folder holds one pair per line:
    {"sources": ["src/0001.jpg"], "target": "tgt/0001.jpg", "instruction": "Remove the lamp."}
with 1-4 sources and paths relative to the folder. The instruction is a plain edit instruction, or a JSON caption of
the result with an "edit_instruction" key (as an object, or that object serialised). A caption file next to the
target (`tgt/0001.txt`, or `tgt/0001.json`) holds the instruction instead when it isn't empty; the sources' caption
files are never read.
"""
from __future__ import annotations

import json
from pathlib import Path


def _json_object(text: str) -> dict | None:
    if not text.lstrip().startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


EDITS_FILE = "edits.jsonl"


def is_edit_dataset(folder: Path) -> bool:
    return (folder / EDITS_FILE).is_file()


def read_edit_rows(folder: Path) -> list[dict]:
    rows = []
    for n, line in enumerate((folder / EDITS_FILE).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{EDITS_FILE} line {n} isn't valid JSON: {e}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{EDITS_FILE} line {n} isn't a JSON object")
        rows.append(row)
    return rows


def write_edit_rows(folder: Path, rows: list[dict]):
    tmp = folder / (EDITS_FILE + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(folder / EDITS_FILE)


def edit_prompt(instruction) -> str:
    """A row's instruction as the prompt: a JSON caption serialised compactly, a plain instruction as is."""
    if isinstance(instruction, dict):
        return json.dumps(instruction, ensure_ascii=False, separators=(",", ":"))
    return str(instruction or "").strip()


CAPTION_EXTS = (".txt", ".json")


def caption_file(folder: Path, row: dict) -> Path | None:
    """The caption file next to the pair's target (target.txt, else target.json), if there is one."""
    target = dataset_path(folder, row.get("target"))
    if target is None:
        return None
    for ext in CAPTION_EXTS:
        path = target.with_suffix(ext)
        if path != target and path.is_file():
            return path
    return None


def pair_prompt(folder: Path, row: dict) -> str:
    """The pair's instruction: its target's caption file when that isn't empty, else the row's "instruction"."""
    path = caption_file(folder, row)
    if path is not None:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            obj = _json_object(text)
            return edit_prompt(obj if obj is not None else text)
    return edit_prompt(row.get("instruction"))


def dataset_path(folder: Path, rel) -> Path | None:
    """A path from edits.jsonl, if it stays inside the dataset folder."""
    if not isinstance(rel, str) or not rel.strip():
        return None
    path = (folder / rel).resolve()
    return path if path.is_relative_to(folder.resolve()) else None


def edit_row_problem(folder: Path, row: dict) -> str | None:
    """Why a row can't be trained on (as Bria's trainer checks it), or None."""
    sources = row.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
        return "needs 1 to 4 sources"
    for rel in sources + [row.get("target")]:
        path = dataset_path(folder, rel)
        if path is None or not path.is_file():
            return f"image not found: {rel}"
    prompt = pair_prompt(folder, row)
    if not prompt:
        return "no instruction (in edits.jsonl or a .txt next to the target)"
    obj = _json_object(prompt)
    if obj is not None:
        if "edit_instruction" not in obj:
            return 'a JSON instruction needs an "edit_instruction" key'
        if "structured_caption" in obj and (not isinstance(obj["structured_caption"], (dict, str)) or len(obj) > 2):
            return 'a "structured_caption" instruction holds only "structured_caption" and "edit_instruction"'
    return None

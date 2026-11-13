"""Append-only JSONL storage for call records."""

import threading
import warnings
from pathlib import Path

from pydantic import ValidationError

from core.llm.records import CallRecord


class Ledger:
    """Appends one CallRecord per line to a JSONL file.

    Each append opens the file, writes one complete line and closes it, so lines already
    written survive a crash. A lock stops concurrent threads from interleaving their lines.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: CallRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def read_records(path: Path) -> list[CallRecord]:
    """Load all records from a JSONL file.

    A final line without a trailing newline that fails to parse is the mark of a crash
    mid-write: it is skipped with a warning. Any other invalid line raises ValueError.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(CallRecord.model_validate_json(line))
        except ValidationError as exc:
            truncated_tail = number == len(lines) and not text.endswith("\n")
            if truncated_tail:
                warnings.warn(f"{path}: skipped truncated final line {number}", stacklevel=2)
                continue
            raise ValueError(f"{path}: invalid record on line {number}") from exc
    return records

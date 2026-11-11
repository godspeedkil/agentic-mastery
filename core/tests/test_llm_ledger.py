import threading

import pytest

from core.llm import CallRecord, Ledger, Usage, read_records


def make_record(n: int = 0) -> CallRecord:
    return CallRecord(provider="anthropic", model="m", usage=Usage(input_tokens=n))


def test_append_creates_directories_and_round_trips(tmp_path):
    path = tmp_path / "runs" / "run-1.jsonl"
    ledger = Ledger(path)
    written = [make_record(n) for n in range(3)]
    for record in written:
        ledger.append(record)
    assert path.read_text().count("\n") == 3
    assert read_records(path) == written


def test_truncated_final_line_is_skipped_with_warning(tmp_path):
    path = tmp_path / "run.jsonl"
    Ledger(path).append(make_record())
    with path.open("a") as f:
        f.write('{"id": "half-writ')  # simulates a crash mid-write
    with pytest.warns(UserWarning, match="truncated"):
        records = read_records(path)
    assert len(records) == 1


def test_invalid_line_in_the_middle_raises(tmp_path):
    path = tmp_path / "run.jsonl"
    ledger = Ledger(path)
    ledger.append(make_record())
    with path.open("a") as f:
        f.write("not a record\n")
    ledger.append(make_record())
    with pytest.raises(ValueError, match="line 2"):
        read_records(path)


def test_concurrent_appends_do_not_interleave(tmp_path):
    path = tmp_path / "run.jsonl"
    ledger = Ledger(path)

    def write_many() -> None:
        for n in range(50):
            ledger.append(make_record(n))

    threads = [threading.Thread(target=write_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(read_records(path)) == 400

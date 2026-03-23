from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPECTED_COMBINED_CORPUS_FILE_COUNT = 33
TEXT_HEADER_CANDIDATES = ("statement", "text", "content", "article", "body")
LABEL_HEADER_CANDIDATES = ("label", "target", "class")
LABEL_NORMALIZATION_MAP = {
    "0": 0,
    "1": 1,
    "fake": 0,
    "false": 0,
    "unreliable": 0,
    "real": 1,
    "true": 1,
    "reliable": 1,
}
LABEL_START_RE = re.compile(r'^\s*["\']?([A-Za-z0-9_\-]+)["\']?\s*,')
WHITESPACE_RE = re.compile(r"\s+")
PANDAS_READ_ATTEMPTS = (
    ("pandas_c", {"low_memory": False}),
    ("pandas_python", {"engine": "python"}),
    (
        "pandas_python_escape",
        {
            "engine": "python",
            "quotechar": '"',
            "escapechar": "\\",
            "doublequote": True,
        },
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_header_token(token) -> str:
    return str(token).strip().strip("\"'").lower().replace("\ufeff", "")


def _label_token(value) -> str | None:
    if pd.isna(value):
        return None
    token = str(value).strip().strip("\"'")
    return token or None


def normalize_binary_label(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer() and int(value) in {0, 1}:
            return int(value)
        return None
    token = _label_token(value)
    if token is None:
        return None
    return LABEL_NORMALIZATION_MAP.get(token.lower())


def clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", "\n").replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text.replace('""', '"').strip()


def _append_dropped_sample(samples: list[dict], *, max_samples: int, **row) -> None:
    if len(samples) >= max_samples:
        return
    samples.append(row)


def _resolve_existing_path(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    direct = Path(relative_path)
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Missing required path: {relative_path}")


def discover_combined_corpus_files(root: Path) -> tuple[list[dict], str]:
    train_dir = _resolve_existing_path(root, "Combined_Corpus_Dataset/Data/Train")
    test_dir = _resolve_existing_path(root, "Combined_Corpus_Dataset/Data/Test")

    file_entries = []
    for split_dir in (train_dir, test_dir):
        for csv_path in sorted(split_dir.glob("*.csv")):
            rel_path = csv_path.relative_to(root) if root in csv_path.parents else csv_path
            file_entries.append(
                {
                    "absolute_path": csv_path,
                    "relative_path": rel_path.as_posix(),
                    "original_split_dir": split_dir.name,
                    "file_size_bytes": int(csv_path.stat().st_size),
                }
            )

    fingerprint = hashlib.sha256(
        json.dumps(
            [
                [entry["relative_path"], entry["file_size_bytes"]]
                for entry in file_entries
            ],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return file_entries, fingerprint


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    normalized = {_normalize_header_token(col): col for col in df.columns}
    text_col = next((normalized[name] for name in TEXT_HEADER_CANDIDATES if name in normalized), None)
    label_col = next((normalized[name] for name in LABEL_HEADER_CANDIDATES if name in normalized), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"could not detect text/label columns from headers: {list(df.columns)}"
        )
    return text_col, label_col


def _record_from_values(text_value, label_value, file_entry: dict) -> dict | None:
    cleaned_text = clean_text_value(text_value)
    normalized_label = normalize_binary_label(label_value)
    if not cleaned_text or normalized_label is None:
        return None
    return {
        "text": cleaned_text,
        "label": int(normalized_label),
        "source_file": file_entry["relative_path"],
        "original_split_dir": file_entry["original_split_dir"],
        "content": cleaned_text,
    }


def _standardize_dataframe(
    df: pd.DataFrame,
    *,
    file_entry: dict,
    parser_method: str,
    dropped_rows_sample: list[dict],
    raw_label_counter: Counter,
    max_dropped_samples: int,
) -> tuple[pd.DataFrame, int, int]:
    text_col, label_col = _detect_columns(df)
    raw_row_count = int(len(df))
    cleaned_records = []
    dropped_count = 0

    for raw_row_number, (text_value, label_value) in enumerate(
        df[[text_col, label_col]].itertuples(index=False, name=None),
        start=1,
    ):
        raw_label = _label_token(label_value)
        if raw_label is not None:
            raw_label_counter[raw_label] += 1

        record = _record_from_values(text_value, label_value, file_entry)
        if record is None:
            dropped_count += 1
            reason = "empty_text_after_cleaning" if not clean_text_value(text_value) else "label_not_recoverable"
            _append_dropped_sample(
                dropped_rows_sample,
                max_samples=max_dropped_samples,
                source_file=file_entry["relative_path"],
                original_split_dir=file_entry["original_split_dir"],
                parser_method=parser_method,
                raw_row_number=raw_row_number,
                reason=reason,
                raw_label=raw_label or "",
                raw_text_excerpt=str(text_value)[:500],
            )
            continue

        cleaned_records.append(record)

    cleaned_df = pd.DataFrame.from_records(
        cleaned_records,
        columns=["text", "label", "source_file", "original_split_dir", "content"],
    )
    return cleaned_df, raw_row_count, dropped_count


def _determine_record_format(header_line: str) -> str:
    parts = [_normalize_header_token(part) for part in header_line.split(",")]
    label_positions = [idx for idx, token in enumerate(parts) if token in LABEL_HEADER_CANDIDATES]
    text_positions = [idx for idx, token in enumerate(parts) if token in TEXT_HEADER_CANDIDATES]
    if not label_positions or not text_positions:
        raise ValueError(f"unrecognized header: {header_line!r}")
    return "label_first" if label_positions[0] < text_positions[0] else "label_last"


def _recover_label_last(buffer_lines: list[str]) -> tuple[str, str] | None:
    candidate = "\n".join(buffer_lines).strip()
    if not candidate or "," not in candidate:
        return None
    raw_text, raw_label = candidate.rsplit(",", 1)
    if normalize_binary_label(raw_label) is None:
        return None
    return raw_text, raw_label


def _recover_label_first(candidate: str) -> tuple[str, str] | None:
    if "," not in candidate:
        return None
    raw_label, raw_text = candidate.split(",", 1)
    if normalize_binary_label(raw_label) is None:
        return None
    return raw_text, raw_label


def _starts_label_first_record(line: str) -> bool:
    match = LABEL_START_RE.match(line)
    return bool(match and normalize_binary_label(match.group(1)) is not None)


def _custom_recovery_load(
    file_entry: dict,
    *,
    dropped_rows_sample: list[dict],
    raw_label_counter: Counter,
    max_dropped_samples: int,
) -> tuple[pd.DataFrame, int, int, list[str]]:
    cleaned_records = []
    warnings = []
    raw_row_count = 0
    dropped_count = 0
    parser_method = "custom_recovery"

    def register_candidate(raw_text, raw_label, raw_row_number: int, reason_prefix: str = "") -> None:
        nonlocal dropped_count
        label_token = _label_token(raw_label)
        if label_token is not None:
            raw_label_counter[label_token] += 1
        record = _record_from_values(raw_text, raw_label, file_entry)
        if record is None:
            dropped_count += 1
            reason = reason_prefix or (
                "empty_text_after_cleaning" if not clean_text_value(raw_text) else "label_not_recoverable"
            )
            _append_dropped_sample(
                dropped_rows_sample,
                max_samples=max_dropped_samples,
                source_file=file_entry["relative_path"],
                original_split_dir=file_entry["original_split_dir"],
                parser_method=parser_method,
                raw_row_number=raw_row_number,
                reason=reason,
                raw_label=label_token or "",
                raw_text_excerpt=str(raw_text)[:500],
            )
            return
        cleaned_records.append(record)

    def register_unrecoverable(buffer_lines: list[str], raw_row_number: int, reason: str) -> None:
        nonlocal dropped_count
        dropped_count += 1
        warnings.append(reason)
        _append_dropped_sample(
            dropped_rows_sample,
            max_samples=max_dropped_samples,
            source_file=file_entry["relative_path"],
            original_split_dir=file_entry["original_split_dir"],
            parser_method=parser_method,
            raw_row_number=raw_row_number,
            reason=reason,
            raw_label="",
            raw_text_excerpt="\n".join(buffer_lines)[:500],
        )

    with open(
        file_entry["absolute_path"],
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        header_line = None
        for line in handle:
            candidate = line.rstrip("\n\r")
            if candidate.strip():
                header_line = candidate
                break
        if header_line is None:
            raise ValueError("file is empty")

        record_format = _determine_record_format(header_line)

        if record_format == "label_last":
            buffer_lines: list[str] = []
            for line in handle:
                buffer_lines.append(line.rstrip("\n\r"))
                recovered = _recover_label_last(buffer_lines)
                if recovered is None:
                    continue
                raw_row_count += 1
                raw_text, raw_label = recovered
                register_candidate(raw_text, raw_label, raw_row_count)
                buffer_lines = []

            if buffer_lines and any(part.strip() for part in buffer_lines):
                raw_row_count += 1
                register_unrecoverable(
                    buffer_lines,
                    raw_row_count,
                    "label_not_recoverable_after_join",
                )
        else:
            buffer_lines = []
            for line in handle:
                current_line = line.rstrip("\n\r")
                if not buffer_lines and not current_line.strip():
                    continue

                if _starts_label_first_record(current_line):
                    if buffer_lines:
                        raw_row_count += 1
                        recovered = _recover_label_first("\n".join(buffer_lines))
                        if recovered is None:
                            register_unrecoverable(
                                buffer_lines,
                                raw_row_count,
                                "label_not_recoverable_after_join",
                            )
                        else:
                            raw_text, raw_label = recovered
                            register_candidate(raw_text, raw_label, raw_row_count)
                    buffer_lines = [current_line]
                    continue

                if not buffer_lines:
                    buffer_lines = [current_line]
                else:
                    buffer_lines.append(current_line)

            if buffer_lines:
                raw_row_count += 1
                recovered = _recover_label_first("\n".join(buffer_lines))
                if recovered is None:
                    register_unrecoverable(
                        buffer_lines,
                        raw_row_count,
                        "label_not_recoverable_after_join",
                    )
                else:
                    raw_text, raw_label = recovered
                    register_candidate(raw_text, raw_label, raw_row_count)

    cleaned_df = pd.DataFrame.from_records(
        cleaned_records,
        columns=["text", "label", "source_file", "original_split_dir", "content"],
    )
    return cleaned_df, raw_row_count, dropped_count, warnings


def _load_single_combined_corpus_file(
    file_entry: dict,
    *,
    dropped_rows_sample: list[dict],
    raw_label_counter: Counter,
    max_dropped_samples: int,
) -> tuple[pd.DataFrame, str, int, int, list[str]]:
    notes = []

    for parser_method, read_kwargs in PANDAS_READ_ATTEMPTS:
        attempt_dropped_rows_sample: list[dict] = []
        attempt_raw_label_counter: Counter = Counter()
        try:
            df = pd.read_csv(
                file_entry["absolute_path"],
                on_bad_lines="error",
                **read_kwargs,
            )
            cleaned_df, raw_row_count, dropped_count = _standardize_dataframe(
                df,
                file_entry=file_entry,
                parser_method=parser_method,
                dropped_rows_sample=attempt_dropped_rows_sample,
                raw_label_counter=attempt_raw_label_counter,
                max_dropped_samples=max_dropped_samples,
            )
            if cleaned_df.empty:
                raise ValueError("no usable rows after cleaning")
            if dropped_count > 0:
                raise ValueError(
                    f"parser produced dropped rows after cleaning ({dropped_count}/{raw_row_count}); "
                    "falling back to stricter recovery"
                )
            dropped_rows_sample.extend(attempt_dropped_rows_sample)
            raw_label_counter.update(attempt_raw_label_counter)
            return cleaned_df, parser_method, raw_row_count, dropped_count, notes
        except Exception as exc:
            notes.append(f"{parser_method} failed: {exc}")

    try:
        attempt_dropped_rows_sample = []
        attempt_raw_label_counter = Counter()
        cleaned_df, raw_row_count, dropped_count, warnings = _custom_recovery_load(
            file_entry,
            dropped_rows_sample=attempt_dropped_rows_sample,
            raw_label_counter=attempt_raw_label_counter,
            max_dropped_samples=max_dropped_samples,
        )
        if cleaned_df.empty:
            raise ValueError("custom recovery produced no usable rows")
        dropped_rows_sample.extend(attempt_dropped_rows_sample)
        raw_label_counter.update(attempt_raw_label_counter)
        notes.extend(warnings)
        return cleaned_df, "custom_recovery", raw_row_count, dropped_count, notes
    except Exception as exc:
        notes.append(f"custom_recovery failed: {exc}")
        raise RuntimeError(" | ".join(notes)) from exc


def _save_cleaned_dataset(cleaned_df: pd.DataFrame, *, parquet_path: Path, csv_path: Path) -> None:
    cleaned_df.to_csv(csv_path, index=False)
    try:
        cleaned_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"failed to save required parquet dataset at {parquet_path}; "
            "install a parquet engine such as pyarrow"
        ) from exc


def _validate_cached_summary(
    summary_path: Path,
    *,
    manifest_fingerprint: str,
    cleaned_parquet_path: Path,
    cleaned_csv_path: Path,
) -> dict | None:
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not summary.get("all_files_processed_successfully"):
        return None
    if summary.get("manifest_fingerprint") != manifest_fingerprint:
        return None
    if not cleaned_csv_path.exists() or not cleaned_parquet_path.exists():
        return None
    return summary


def _load_cached_cleaned_dataset(cleaned_parquet_path: Path, cleaned_csv_path: Path) -> pd.DataFrame:
    try:
        cached = pd.read_parquet(cleaned_parquet_path)
    except Exception:
        cached = pd.read_csv(cleaned_csv_path)
    if "content" not in cached.columns and "text" in cached.columns:
        cached["content"] = cached["text"].astype(str)
    return cached


def print_combined_corpus_audit_summary(summary: dict) -> None:
    parser_usage = summary.get("parser_methods", {})
    parser_text = ", ".join(f"{name}={count}" for name, count in sorted(parser_usage.items())) or "none"
    print("Combined_Corpus audit summary")
    print(
        f"  files: {summary.get('discovered_file_count')} discovered / "
        f"{summary.get('expected_file_count')} expected"
    )
    print(f"  raw rows: {summary.get('total_raw_rows')}")
    print(f"  cleaned rows: {summary.get('total_cleaned_rows')}")
    print(f"  dropped rows: {summary.get('total_dropped_rows')}")
    print(f"  parser usage: {parser_text}")
    print(f"  manifest: {summary.get('manifest_path')}")
    print(f"  summary: {summary.get('summary_path')}")
    print(f"  cleaned csv: {summary.get('cleaned_csv_path')}")
    print(f"  cleaned parquet: {summary.get('cleaned_parquet_path')}")


def load_combined_corpus_dataset(
    root: Path,
    *,
    expected_file_count: int = EXPECTED_COMBINED_CORPUS_FILE_COUNT,
    max_dropped_samples: int = 250,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    root = Path(root)
    data_dir = _resolve_existing_path(root, "Combined_Corpus_Dataset/Data")
    results_dir = root / "results" / "combined_corpus"
    results_dir.mkdir(parents=True, exist_ok=True)

    cleaned_parquet_path = data_dir / "combined_corpus_cleaned.parquet"
    cleaned_csv_path = data_dir / "combined_corpus_cleaned.csv"
    manifest_path = results_dir / "data_audit_manifest.csv"
    summary_path = results_dir / "data_audit_summary.json"
    dropped_rows_sample_path = results_dir / "dropped_rows_sample.csv"

    file_entries, manifest_fingerprint = discover_combined_corpus_files(root)
    cached_summary = None if force_rebuild else _validate_cached_summary(
        summary_path,
        manifest_fingerprint=manifest_fingerprint,
        cleaned_parquet_path=cleaned_parquet_path,
        cleaned_csv_path=cleaned_csv_path,
    )
    if cached_summary is not None:
        dataset = _load_cached_cleaned_dataset(cleaned_parquet_path, cleaned_csv_path)
        dataset.attrs["audit_summary"] = cached_summary
        print("Reusing cached cleaned Combined_Corpus dataset")
        print_combined_corpus_audit_summary(cached_summary)
        print(f"Combined_Corpus final usable rows: {len(dataset)}")
        return dataset

    dropped_rows_sample: list[dict] = []
    raw_label_counter: Counter = Counter()
    parser_counter: Counter = Counter()
    manifest_rows = []
    cleaned_frames = []
    failed_files = []

    for file_entry in file_entries:
        try:
            cleaned_df, parser_method, raw_row_count, dropped_count, notes = _load_single_combined_corpus_file(
                file_entry,
                dropped_rows_sample=dropped_rows_sample,
                raw_label_counter=raw_label_counter,
                max_dropped_samples=max_dropped_samples,
            )
            parser_counter[parser_method] += 1
            cleaned_frames.append(cleaned_df)
            manifest_rows.append(
                {
                    "file_path": file_entry["relative_path"],
                    "file_size_bytes": file_entry["file_size_bytes"],
                    "load_status": "loaded",
                    "parser_method": parser_method,
                    "raw_row_count": int(raw_row_count),
                    "cleaned_row_count": int(len(cleaned_df)),
                    "dropped_row_count": int(dropped_count),
                    "error_message": " | ".join(notes),
                }
            )
        except Exception as exc:
            failed_files.append(file_entry["relative_path"])
            manifest_rows.append(
                {
                    "file_path": file_entry["relative_path"],
                    "file_size_bytes": file_entry["file_size_bytes"],
                    "load_status": "failed",
                    "parser_method": "",
                    "raw_row_count": 0,
                    "cleaned_row_count": 0,
                    "dropped_row_count": 0,
                    "error_message": str(exc),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("file_path").reset_index(drop=True)
    manifest_df.to_csv(manifest_path, index=False)

    dropped_rows_df = pd.DataFrame(
        dropped_rows_sample,
        columns=[
            "source_file",
            "original_split_dir",
            "parser_method",
            "raw_row_number",
            "reason",
            "raw_label",
            "raw_text_excerpt",
        ],
    )
    dropped_rows_df.to_csv(dropped_rows_sample_path, index=False)

    total_cleaned_rows = int(manifest_df["cleaned_row_count"].sum()) if not manifest_df.empty else 0
    total_raw_rows = int(manifest_df["raw_row_count"].sum()) if not manifest_df.empty else 0
    total_dropped_rows = int(manifest_df["dropped_row_count"].sum()) if not manifest_df.empty else 0
    label_value_counts = {key: int(value) for key, value in sorted(raw_label_counter.items())}
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "manifest_fingerprint": manifest_fingerprint,
                "total_cleaned_rows": total_cleaned_rows,
                "total_dropped_rows": total_dropped_rows,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "expected_file_count": int(expected_file_count),
        "discovered_file_count": int(len(file_entries)),
        "all_files_discovered": len(file_entries) == expected_file_count,
        "successful_file_count": int((manifest_df["load_status"] == "loaded").sum()) if not manifest_df.empty else 0,
        "failed_file_count": int((manifest_df["load_status"] == "failed").sum()) if not manifest_df.empty else 0,
        "failed_files": failed_files,
        "manifest_fingerprint": manifest_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "total_raw_rows": total_raw_rows,
        "total_cleaned_rows": total_cleaned_rows,
        "total_dropped_rows": total_dropped_rows,
        "parser_methods": {key: int(value) for key, value in sorted(parser_counter.items())},
        "raw_label_value_counts": label_value_counts,
        "label_normalization": {key: int(value) for key, value in sorted(LABEL_NORMALIZATION_MAP.items())},
        "manifest_path": manifest_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "cleaned_csv_path": cleaned_csv_path.as_posix(),
        "cleaned_parquet_path": cleaned_parquet_path.as_posix(),
        "dropped_rows_sample_path": dropped_rows_sample_path.as_posix(),
        "all_files_processed_successfully": False,
    }

    if len(file_entries) != expected_file_count or failed_files:
        _json_dump(summary_path, summary)
        problems = []
        if len(file_entries) != expected_file_count:
            problems.append(
                f"expected {expected_file_count} CSV shards but discovered {len(file_entries)}"
            )
        if failed_files:
            problems.append(f"failed files: {failed_files}")
        raise RuntimeError(
            "Combined_Corpus ingestion audit failed. "
            + "; ".join(problems)
            + f". See {manifest_path} and {summary_path}."
        )

    cleaned_dataset = pd.concat(cleaned_frames, ignore_index=True)
    _save_cleaned_dataset(
        cleaned_dataset,
        parquet_path=cleaned_parquet_path,
        csv_path=cleaned_csv_path,
    )

    summary["all_files_processed_successfully"] = True
    _json_dump(summary_path, summary)

    cleaned_dataset.attrs["audit_summary"] = summary
    print_combined_corpus_audit_summary(summary)
    print(f"Combined_Corpus final usable rows: {len(cleaned_dataset)}")
    return cleaned_dataset

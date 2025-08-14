"""IO utilities for reading and writing data files."""

from pathlib import Path
from typing import Any, Dict, Iterator, List

import orjson
import pandas as pd
from pydantic import BaseModel


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read a JSONL file and yield records.

    Args:
        path: Path to the JSONL file

    Returns:
        Iterator of dictionary records

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}")


def write_jsonl(path: str | Path, records: List[Dict[str, Any] | BaseModel]) -> None:
    """Write records to a JSONL file.

    Args:
        path: Path to write the JSONL file
        records: List of records (dicts or Pydantic models)

    Raises:
        IOError: If unable to write to the file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        for record in records:
            if isinstance(record, BaseModel):
                data = record.model_dump()
            else:
                data = record
            f.write(orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE))


def write_parquet(path: str | Path, records: List[Dict[str, Any] | BaseModel]) -> None:
    """Write records to a Parquet file.

    Args:
        path: Path to write the Parquet file
        records: List of records (dicts or Pydantic models)

    Raises:
        IOError: If unable to write to the file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for record in records:
        if isinstance(record, BaseModel):
            data.append(record.model_dump())
        else:
            data.append(record)

    df = pd.DataFrame(data)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

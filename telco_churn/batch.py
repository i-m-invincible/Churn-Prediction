"""Bounded CSV input and spreadsheet-safe prediction exports."""

from io import BytesIO

import pandas as pd

MAX_UPLOAD_BYTES = 5_000_000
MAX_BATCH_ROWS = 10_000


def read_customer_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Please upload a CSV smaller than 5 MB.")
    try:
        frame = pd.read_csv(BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header row.") from exc
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError("Please upload at most 10,000 customers per batch.")
    return frame


def prediction_csv(frame: pd.DataFrame) -> bytes:
    """Escape untrusted customer IDs when exporting to spreadsheet applications."""
    output = frame.copy()
    if "customerID" in output:

        def escape(value):
            text = str(value)
            if text.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + text
            return text

        output["customerID"] = output["customerID"].map(escape)
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8")

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from telco_churn.batch import MAX_BATCH_ROWS, MAX_UPLOAD_BYTES, prediction_csv, read_customer_csv
from telco_churn.inference import predict_customers
from telco_churn.schema import ROOT


def test_example_batch_and_csv_export_match_saved_model(bundle):
    frame = read_customer_csv((ROOT / "examples" / "customer_batch.csv").read_bytes())
    output = predict_customers(bundle, frame)
    output.insert(0, "customerID", frame["customerID"])
    restored = pd.read_csv(BytesIO(prediction_csv(output)))
    assert len(restored) == 3
    assert restored.customerID.tolist() == frame.customerID.tolist()
    np.testing.assert_allclose(restored.churn_probability, output.churn_probability)


@pytest.mark.parametrize("content", [b"", b'bad,header\n"unclosed', b"header\n\xff"])
def test_invalid_csv_has_a_friendly_error(content):
    with pytest.raises(ValueError, match="valid UTF-8 CSV"):
        read_customer_csv(content)


def test_batch_limits_are_checked():
    with pytest.raises(ValueError, match="5 MB"):
        read_customer_csv(b"a" * (MAX_UPLOAD_BYTES + 1))
    content = b"header\n" + b"value\n" * (MAX_BATCH_ROWS + 1)
    with pytest.raises(ValueError, match="10,000"):
        read_customer_csv(content)


def test_spreadsheet_formulas_in_customer_ids_are_escaped():
    output = pd.DataFrame(
        {
            "customerID": ["=1+1", "  +1+1", "@SUM(1,2)", "EXAMPLE-01"],
            "churn_probability": [0.1, 0.2, 0.3, 0.4],
        }
    )
    restored = pd.read_csv(BytesIO(prediction_csv(output)))
    assert restored.customerID.tolist() == ["'=1+1", "'  +1+1", "'@SUM(1,2)", "EXAMPLE-01"]
    assert output.customerID.iloc[0] == "=1+1"

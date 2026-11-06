"""Check artifact integrity and agreement between saved components and pipeline."""

import joblib
import numpy as np

from telco_churn.data import load_dataset
from telco_churn.inference import load_bundle, predict_customers
from telco_churn.schema import ARTIFACT_DIR


def main():
    bundle = load_bundle()
    features, _, _ = load_dataset()
    sample = features.head(20)
    predictions = predict_customers(bundle, sample)
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    standalone = model.predict_proba(preprocessor.transform(sample))[:, 1]
    np.testing.assert_allclose(predictions["churn_probability"], standalone, rtol=1e-6)
    print(
        f"Verified {bundle.metadata['selected_model']}: hashes, versions, schema, "
        "probabilities and component/pipeline agreement."
    )


if __name__ == "__main__":
    main()

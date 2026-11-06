import pytest

from telco_churn.data import load_dataset
from telco_churn.inference import load_bundle


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def bundle():
    return load_bundle()

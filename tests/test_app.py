import pytest
from streamlit.testing.v1 import AppTest

from telco_churn.schema import ROOT


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def test_initial_prediction_and_submission(app):
    assert not app.exception
    assert not app.error
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].label == "Estimated churn probability"
    assert app.metric[0].value.endswith("%")
    assert "assessment" in app.session_state


def test_missing_charges_and_no_services_are_handled(app):
    app.selectbox[0].select("No").run()
    app.selectbox[1].select("No").run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    raw = app.session_state["assessment"][2]
    assert raw["TotalCharges"] is None
    assert raw["MultipleLines"] == "No phone service"
    assert raw["OnlineSecurity"] == "No internet service"


@pytest.mark.parametrize("page", ["Explore data", "Model performance"])
def test_dashboard_pages(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) >= 1

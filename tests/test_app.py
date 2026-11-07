import pytest
from streamlit.testing.v1 import AppTest

from telco_churn.schema import ROOT


def button(app, label):
    return next(item for item in app.button if item.label == label)


def selectbox(app, key):
    return next(item for item in app.selectbox if item.key == key)


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def test_initial_prediction_and_submission(app):
    assert not app.exception
    assert not app.error
    button(app, "Predict churn").click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].label == "Estimated churn probability"
    assert app.metric[0].value.endswith("%")
    assert "assessment" in app.session_state


def test_missing_charges_and_no_services_are_handled(app):
    selectbox(app, "customer_PhoneService").select("No").run()
    selectbox(app, "customer_InternetService").select("No").run()
    app.checkbox[0].check().run()
    button(app, "Predict churn").click().run()
    assert not app.exception
    assert not app.error
    raw = app.session_state["assessment"][2]
    assert raw["TotalCharges"] is None
    assert raw["MultipleLines"] == "No phone service"
    assert raw["OnlineSecurity"] == "No internet service"


@pytest.mark.parametrize("page", ["Explore Data", "Model Performance"])
def test_dashboard_pages(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) >= 1


def test_examples_recompute_real_predictions(app):
    original = app.session_state["assessment"][0]["churn_probability"]
    button(app, "Established customer").click().run()
    assert not app.exception and not app.error
    result, _, raw = app.session_state["assessment"]
    assert raw["tenure"] == 60
    assert result["churn_probability"] != original
    button(app, "New subscriber").click().run()
    assert app.session_state["assessment"][0]["churn_probability"] == pytest.approx(original)


def test_resource_failures_do_not_reveal_internal_paths(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("/workspace/private/internal-file secret-placeholder")

    monkeypatch.setattr("telco_churn.dashboard.resources", broken)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "temporarily unavailable" in app.error[0].value
    assert "/workspace" not in app.error[0].value
    assert "secret-placeholder" not in app.error[0].value


def test_batch_view_and_download_with_sample_csv(monkeypatch):
    from io import BytesIO

    content = (ROOT / "examples" / "customer_batch.csv").read_bytes()
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: BytesIO(content))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    scored = next(item for item in app.metric if item.label == "Customers scored")
    assert scored.value == "3"
    assert any("Download predictions" in str(item.proto) for item in app.get("download_button"))

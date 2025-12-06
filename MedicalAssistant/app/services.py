import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ART_DIR = BASE_DIR / "ml" / "artifacts"
PLOT_DIR = BASE_DIR / "ml" / "plots"

model = joblib.load(ART_DIR / "xgb_model.pkl")

def predict_risk(data_dict):
    X_user = pd.DataFrame([data_dict])
    prob = float(model.predict_proba(X_user)[0, 1])
    return prob

def risk_gauge_html(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30]},
                {"range": [30, 70]},
                {"range": [70, 100]},
            ]
        }
    ))
    return fig.to_html(full_html=False)

def load_plot_html(name):
    p = PLOT_DIR / f"{name}.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "<div>Plot not found</div>"

def shap_individual_html(data_dict):
    X_user = pd.DataFrame([data_dict])
    preprocess = model.named_steps["preprocess"]
    xgb = model.named_steps["model"]
    X_trans = preprocess.transform(X_user)
    feature_names = preprocess.get_feature_names_out()
    background = np.load(ART_DIR / "shap_background.npy")
    explainer = shap.TreeExplainer(xgb, background)
    shap_values = explainer.shap_values(X_trans)
    fig = shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], feature_names=feature_names, show=False
    )
    import matplotlib.pyplot as plt
    import io, base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"

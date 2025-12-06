import json
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "health_risk_dataset.xlsx"
ART_DIR = BASE_DIR / "ml" / "artifacts"
PLOT_DIR = BASE_DIR / "ml" / "plots"
ART_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

num_cols = ["Age", "BMI", "GlucoseLevel", "BloodPressure", "FamilyHistory"]
cat_cols = ["ExerciseLevel"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42
    )
}

results = {}
roc_traces = []

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    fpr, tpr, thr = roc_curve(y_test, proba)
    roc_traces.append(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC={auc:.3f}"))

    results[name] = {
        "AUC": float(auc),
        "ACC": float(acc),
        "CM": cm.tolist()
    }

fig_roc = go.Figure(roc_traces + [go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash"))])
fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="FPR", yaxis_title="TPR")
fig_roc.write_html(PLOT_DIR / "roc_curve.html")

with open(ART_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

xgb_pipe = Pipeline([("preprocess", preprocess), ("model", models["XGBoost"])])
xgb_pipe.fit(X_train, y_train)

joblib.dump(xgb_pipe, ART_DIR / "xgb_model.pkl")

feature_names = xgb_pipe.named_steps["preprocess"].get_feature_names_out()
importances = xgb_pipe.named_steps["model"].feature_importances_

fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values("importance", ascending=False)

fig_fi = px.bar(fi.head(12), x="importance", y="feature", orientation="h", title="Feature Importance (XGBoost)")
fig_fi.write_html(PLOT_DIR / "feature_importance.html")

with open(ART_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
    json.dump(fi.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

risk_counts = df["Outcome"].value_counts()
fig_pie = go.Figure(data=[go.Pie(labels=["Healthy", "Diabetic"], values=risk_counts.values)])
fig_pie.update_layout(title="Population Risk Distribution")
fig_pie.write_html(PLOT_DIR / "population_pie.html")

corr = df[num_cols + ["Outcome"]].corr()
fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
fig_corr.write_html(PLOT_DIR / "corr_heatmap.html")
with open(ART_DIR / "corr_matrix.json", "w", encoding="utf-8") as f:
    json.dump(corr.to_dict(), f, ensure_ascii=False, indent=2)

df["AgeGroup"] = pd.cut(df["Age"], bins=[0,30,45,60,100])
trend = df.groupby("AgeGroup")["Outcome"].mean().reset_index()
trend["AgeGroup"] = trend["AgeGroup"].astype(str)
fig_trend = px.line(trend, x="AgeGroup", y="Outcome", markers=True, title="Risk Trend by Age Groups")
fig_trend.write_html(PLOT_DIR / "risk_trend.html")

X_train_trans = xgb_pipe.named_steps["preprocess"].transform(X_train)
background = shap.sample(X_train_trans, 200, random_state=42)
np.save(ART_DIR / "shap_background.npy", background)

explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"], background)
shap_values = explainer.shap_values(X_train_trans)

comp_df = pd.DataFrame({
    "Model": list(results.keys()),
    "AUC": [results[m]["AUC"] for m in results],
    "ACC": [results[m]["ACC"] for m in results]
})
fig_cmp = px.bar(comp_df, x="Model", y=["AUC","ACC"], barmode="group", title="Model Comparison")
fig_cmp.write_html(PLOT_DIR / "comparison.html")

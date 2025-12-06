from django.shortcuts import render, redirect
from .forms import RiskForm
from .services import predict_risk, risk_gauge_html, load_plot_html, shap_individual_html

def form_view(request):
    form = RiskForm()
    return render(request, "form.html", {"form": form})

def result_view(request):
    if request.method != "POST":
        return redirect("form")

    form = RiskForm(request.POST)
    if not form.is_valid():
        return render(request, "form.html", {"form": form})

    data = {
        "Age": form.cleaned_data["age"],
        "BMI": form.cleaned_data["bmi"],
        "GlucoseLevel": form.cleaned_data["glucose"],
        "BloodPressure": form.cleaned_data["bp"],
        "FamilyHistory": int(form.cleaned_data["family"]),
        "ExerciseLevel": form.cleaned_data["exercise"],
    }

    prob = predict_risk(data)
    gauge = risk_gauge_html(prob)
    shap_html = shap_individual_html(data)

    return render(request, "result.html", {
        "prob": round(prob * 100, 1),
        "gauge": gauge,
        "shap_html": shap_html
    })

def admin_dashboard(request):
    plots = {
        "roc": load_plot_html("roc_curve"),
        "fi": load_plot_html("feature_importance"),
        "cmp": load_plot_html("comparison"),
        "pie": load_plot_html("population_pie"),
        "corr": load_plot_html("corr_heatmap"),
        "trend": load_plot_html("risk_trend"),
    }
    return render(request, "admin_dashboard.html", plots)

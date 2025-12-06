from django import forms

class RiskForm(forms.Form):
    age = forms.FloatField(min_value=0, max_value=120)
    bmi = forms.FloatField(min_value=10, max_value=80)
    glucose = forms.FloatField(min_value=40, max_value=400)
    bp = forms.FloatField(min_value=40, max_value=250)
    family = forms.ChoiceField(choices=[(0, "No"), (1, "Yes")])
    exercise = forms.ChoiceField(choices=[("Low", "Low"), ("Medium", "Medium"), ("High", "High")])

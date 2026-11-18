from pathlib import Path
import joblib
import pandas as pd

FEATURES = ['monthly_income', 'age', 'savings_goal', 'food_expense',
            'travel_expense', 'shopping_expense', 'bills_expense',
            'previous_month_expense']

def load_model(path):
    path = Path(path)
    return joblib.load(path) if path.exists() else None

def predict_expense(bundle, values):
    row = pd.DataFrame([{name: values[name] for name in FEATURES}])
    return float(bundle['model'].predict(row)[0])

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA = ROOT / 'data' / 'personal_expense_dataset.csv'
MODEL = ROOT / 'models' / 'expense_model.pkl'
FEATURES = ['monthly_income', 'age', 'savings_goal', 'food_expense',
            'travel_expense', 'shopping_expense', 'bills_expense',
            'previous_month_expense']
TARGET = 'next_month_expense'

df = pd.read_csv(DATA)
print('Dataset shape:', df.shape)
print('Missing values:\n', df.isnull().sum())
print('Duplicate rows:', df.duplicated().sum())
df = df.drop_duplicates().dropna(subset=FEATURES + [TARGET])
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

candidates = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42
    )
}
results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'mae': mean_absolute_error(y_test, pred),
        'rmse': mean_squared_error(y_test, pred) ** 0.5,
        'r2_score': r2_score(y_test, pred)
    }
    print(f"\n{name}: MAE={results[name]['mae']:.2f}, "
          f"RMSE={results[name]['rmse']:.2f}, R2={results[name]['r2_score']:.4f}")

best_name = min(results, key=lambda key: results[key]['mae'])
best = results[best_name]
cv_scores = cross_val_score(best['model'], X, y, cv=5, scoring='neg_mean_absolute_error')
print('Selected model:', best_name)
print('5-fold CV MAE:', round(-cv_scores.mean(), 2))

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    'model': best['model'], 'model_name': best_name,
    'mae': best['mae'], 'rmse': best['rmse'],
    'r2_score': best['r2_score'], 'features': FEATURES
}, MODEL)
print('Saved model:', MODEL)

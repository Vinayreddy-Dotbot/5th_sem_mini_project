# Personal Expense Analyzer

A learning project built with Python, Streamlit, SQLite, Pandas and scikit-learn.

## What's new in v2
- **User login**: create an account with your phone number and a password, then log in or log out.
  Passwords are stored as salted PBKDF2-SHA256 hashes (Python standard library, no extra packages).
- **New interface**: colour theme, a login screen, a sidebar with your profile, metric cards and
  interactive charts with the same colour for each category everywhere.
- **Dashboard**: this month's spending, how much of your budget is used, a category donut and recent transactions.
- **Expenses**: the subcategory list now updates as soon as you change the category. Search and filter
  your expenses, and delete one by picking it from a list instead of typing its ID.
- **Analytics**: choose a period (this month, last 3/6/12 months, all time), view top subcategories and a monthly trend by category.
- **Prediction**: inputs are pre-filled from last month's real expenses. The month total is calculated
  for you, the same way it is in the training data. If the model is missing, a button trains it.
- **Budget Advisor**: now uses **this month only** (previously it used all transactions ever saved). It shows how
  much you can safely spend per day and gives a tip for your biggest category.
- **Reports**: a monthly summary (month × category) plus CSV downloads.
- **Profile**: edit your name, income and savings goal, and change your password.

## Files
- `app.py`: Streamlit application (login, Dashboard, Expenses, Analytics, Prediction, Budget Advisor, Reports, Profile)
- `auth.py`: password hashing and checking
- `database.py`: SQLite tables and database operations (adds the password column to older databases automatically)
- `train_model.py`: dataset checks, Linear Regression/Random Forest comparison, evaluation and model saving
- `ml_utils.py`: model loading and prediction helper
- `.streamlit/config.toml`: colour theme
- `data/personal_expense_dataset.csv`: synthetic training dataset
- `models/`: the trained model is saved here

## Setup
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python train_model.py        # optional, the Prediction page has a "Train" button too
streamlit run app.py
```

## Existing data
Your old `data/expenses.db` still works. Profiles created before login existed have no password yet:
choose **Create account** and use the same phone number to claim the profile and keep its expenses.

## Notes
- The included CSV is synthetic practice data, not real customer financial data.
- Predictions are for learning and are not financial advice.
- This is a local, single-machine app. Anyone who knows the phone number of an old (unclaimed) profile could
  claim it, so claim yours first. For a public deployment you would also add rate limiting on login.
- Prediction timestamps are stored in UTC.
- The model expects monthly summary inputs with the same definitions as the training CSV.

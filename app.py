import calendar
import html
import subprocess
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import auth
import database
from ml_utils import load_model, predict_expense

st.set_page_config(page_title='Personal Expense Analyzer', page_icon='💰', layout='wide')

ROOT = Path(__file__).parent
DB = ROOT / 'data' / 'expenses.db'
MODEL_PATH = ROOT / 'models' / 'expense_model.pkl'
database.create_tables(DB)

CATEGORIES = {
    'Food': ['Groceries', 'Restaurant', 'Fast food', 'Delivery', 'Other'],
    'Travel': ['Bus', 'Auto', 'Bike', 'Car', 'Train', 'Other'],
    'Shopping': ['Clothing', 'Cosmetics', 'Electronics', 'Household', 'Other'],
    'Bills': ['Electricity', 'Water', 'Internet', 'Rent', 'Maintenance', 'Other'],
    'Healthcare': ['Medicine', 'Doctor', 'Tests', 'Other'],
    'Education': ['Fees', 'Books', 'Courses', 'Other'],
    'Other': ['Miscellaneous'],
}
COLORS = {  # one colour per category, used in every chart
    'Food': '#F97316', 'Travel': '#0EA5E9', 'Shopping': '#A855F7', 'Bills': '#EF4444',
    'Healthcare': '#10B981', 'Education': '#EAB308', 'Other': '#64748B',
}
TIPS = {
    'Food': 'Plan meals for the week and order in less often. Delivery fees and tips add up fast.',
    'Travel': 'Combine errands into one trip, and use the bus, metro or a monthly pass where you can.',
    'Shopping': 'Wait 48 hours before any non-essential purchase. Many of those urges pass.',
    'Bills': 'Cancel subscriptions you no longer use, and compare internet and power plans once a year.',
    'Healthcare': 'Ask for generic medicines and check what your insurance already covers.',
    'Education': 'Look for free or employer-sponsored courses before paying for new ones.',
    'Other': 'Add a note to these expenses. Spending you cannot name is the easiest to cut.',
}
PAGES = ['🏠 Dashboard', '🧾 Expenses', '📊 Analytics', '🤖 Prediction',
         '💡 Budget Advisor', '🗄️ Reports', '👤 Profile']

st.markdown('''<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: #fff;
       padding: 22px 28px; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff !important; font-size: 1.75rem !important; margin: 0 !important; padding: 0 !important;}
.hero p {color: rgba(255,255,255,.85) !important; margin: 4px 0 0 0 !important;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #E5E7EB; border-radius: 14px;
       padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06);}
div[data-testid="stMetricLabel"] p {color: #64748B; font-weight: 600;}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {border-radius: 10px; font-weight: 600;}
.user-card {display: flex; gap: 12px; align-items: center; padding: 12px; background: #fff;
       border: 1px solid #E5E7EB; border-radius: 14px; margin-bottom: 1rem;}
.avatar {width: 42px; height: 42px; border-radius: 50%; background: #4F46E5; color: #fff;
       display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem;}
.user-name {font-weight: 700;}
.user-meta {color: #64748B; font-size: .85rem;}
.txn {display: flex; justify-content: space-between; align-items: center; padding: 9px 0;
       border-bottom: 1px solid #EEF0F4;}
.pill {color: #fff; font-size: .75rem; font-weight: 600; padding: 2px 9px; border-radius: 999px; margin-right: 8px;}
.txn-detail {color: #334155;}
.txn-right {text-align: right;}
.txn-right small {display: block; color: #94A3B8;}
.login-hero {text-align: center; margin: 2.5rem 0 1.5rem 0;}
.login-hero .logo {font-size: 3rem;}
.login-hero h1 {margin: 0 !important; padding: 0 !important;}
.login-hero p {color: #64748B !important;}
</style>''', unsafe_allow_html=True)


# ---------- small helpers ----------

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def money(value):
    value = float(value)
    return f"{'-' if value < 0 else ''}₹{abs(value):,.0f}"


def header(title, subtitle=''):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def flash(message):
    """Show a toast after the next rerun."""
    st.session_state.flash = message


def set_current_user(user):
    st.session_state.user = {k: v for k, v in user.items() if k != 'password_hash'}


def clean_phone(phone):
    return phone.replace(' ', '').replace('-', '').strip()


def load_expenses(user_id):
    rows = database.get_expenses(DB, user_id)
    df = pd.DataFrame(rows, columns=['id', 'date', 'category', 'subcategory', 'amount', 'note'])
    df['date'] = pd.to_datetime(df['date'])
    df[['subcategory', 'note']] = df[['subcategory', 'note']].fillna('')
    return df


def month_slice(df, months_ago=0):
    """Expenses from this month (0), last month (1), and so on."""
    period = pd.Timestamp.today().to_period('M') - months_ago
    return df[df['date'].dt.to_period('M') == period]


# ---------- charts ----------

def color_scale():
    return alt.Scale(domain=list(COLORS), range=list(COLORS.values()))


def donut_chart(df):
    data = df.groupby('category', as_index=False)['amount'].sum()
    chart = alt.Chart(data).mark_arc(innerRadius=70, stroke='white', strokeWidth=2).encode(
        theta='amount:Q',
        color=alt.Color('category:N', scale=color_scale(), legend=alt.Legend(title=None)),
        tooltip=['category:N', alt.Tooltip('amount:Q', title='Amount (₹)', format=',.0f')])
    st.altair_chart(chart.properties(height=300))


def subcategory_chart(df):
    data = df.groupby(['category', 'subcategory'], as_index=False)['amount'].sum().nlargest(8, 'amount')
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('amount:Q', title='Amount (₹)'),
        y=alt.Y('subcategory:N', sort='-x', title=None),
        color=alt.Color('category:N', scale=color_scale(), legend=None),
        tooltip=['category:N', 'subcategory:N',
                 alt.Tooltip('amount:Q', title='Amount (₹)', format=',.0f')])
    st.altair_chart(chart.properties(height=300))


def monthly_chart(df):
    data = (df.assign(month=df['date'].dt.to_period('M').dt.to_timestamp())
              .groupby(['month', 'category'], as_index=False)['amount'].sum())
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('yearmonth(month):O', title=None),
        y=alt.Y('amount:Q', title='Amount (₹)'),
        color=alt.Color('category:N', scale=color_scale(), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip('yearmonth(month):T', title='Month'), 'category:N',
                 alt.Tooltip('amount:Q', title='Amount (₹)', format=',.0f')])
    st.altair_chart(chart.properties(height=320))


def recent_transactions(df):
    rows = []
    for r in df.itertuples():
        color = COLORS.get(r.category, '#64748B')
        detail = html.escape(r.note or r.subcategory)
        rows.append(f"<div class='txn'><div><span class='pill' style='background:{color}'>"
                    f"{html.escape(r.category)}</span><span class='txn-detail'>{detail}</span></div>"
                    f"<div class='txn-right'><b>{money(r.amount)}</b><small>{r.date:%d %b}</small></div></div>")
    st.markdown(''.join(rows), unsafe_allow_html=True)


# ---------- login / sign-up ----------

def login_screen():
    st.markdown("<div class='login-hero'><div class='logo'>💰</div><h1>Personal Expense Analyzer</h1>"
                "<p>Track your spending, see where your money goes and plan your savings.</p></div>",
                unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle:
        login_tab, signup_tab = st.tabs(['Log in', 'Create account'])

        with login_tab:
            with st.form('login'):
                phone = st.text_input('Phone number', key='login_phone')
                password = st.text_input('Password', type='password', key='login_password')
                submitted = st.form_submit_button('Log in', type='primary', width='stretch')
            if submitted:
                user = database.get_user_by_phone(DB, clean_phone(phone))
                if user and auth.verify_password(password, user['password_hash']):
                    set_current_user(user)
                    flash(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error('Wrong phone number or password.')

        with signup_tab:
            with st.form('signup'):
                name = st.text_input('Full name')
                phone = st.text_input('Phone number', key='signup_phone',
                                      placeholder='10-digit mobile number')
                c1, c2 = st.columns(2)
                password = c1.text_input('Password', type='password', key='signup_password',
                                         help='At least 8 characters')
                confirm = c2.text_input('Confirm password', type='password')
                c3, c4 = st.columns(2)
                income = c3.number_input('Monthly income (₹)', min_value=0.0, step=1000.0)
                goal = c4.number_input('Monthly savings goal (₹)', min_value=0.0, step=500.0)
                created = st.form_submit_button('Create account', type='primary', width='stretch')
            if created:
                phone = clean_phone(phone)
                existing = database.get_user_by_phone(DB, phone)
                if not name.strip():
                    st.error('Please enter your name.')
                elif existing and existing['password_hash']:
                    st.error('This phone number already has an account. Please log in.')
                elif not existing and not (phone.isdigit() and len(phone) == 10):
                    st.error('Enter a valid 10-digit phone number.')
                elif len(password) < 8:
                    st.error('Password must be at least 8 characters.')
                elif password != confirm:
                    st.error('Passwords do not match.')
                else:
                    user = database.create_user(DB, name.strip(), phone, auth.hash_password(password),
                                                income, goal)
                    set_current_user(user)
                    flash(f"Account created. Welcome, {user['name']}!")
                    st.rerun()
        st.caption('🔒 Passwords are stored as salted hashes, never as plain text.')


if 'user' not in st.session_state:
    st.session_state.user = None
if message := st.session_state.pop('flash', None):
    st.toast(message, icon='✅')
if st.session_state.user is None:
    login_screen()
    st.stop()

# ---------- everything below needs a logged-in user ----------

user = st.session_state.user
income = float(user['monthly_income'])
goal = float(user['savings_goal'])
df = load_expenses(user['user_id'])
this_month = month_slice(df)
last_month = month_slice(df, 1)
today = date.today()

with st.sidebar:
    st.markdown(f"<div class='user-card'><div class='avatar'>{html.escape(user['name'][:1].upper())}</div>"
                f"<div><div class='user-name'>{html.escape(user['name'])}</div>"
                f"<div class='user-meta'>📱 {html.escape(user['phone'])}</div></div></div>",
                unsafe_allow_html=True)
    page = st.radio('Menu', PAGES, label_visibility='collapsed')
    st.divider()
    if st.button('🚪 Log out', width='stretch'):
        st.session_state.user = None
        flash('You have been logged out.')
        st.rerun()
    st.caption('Personal Expense Analyzer')


if page == '🏠 Dashboard':
    header(f"Hi, {html.escape(user['name'].split()[0])} 👋",
           f"Here is your money snapshot for {today:%B %Y}")
    spent = this_month['amount'].sum()
    spent_last = last_month['amount'].sum()
    allowance = income - goal
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Monthly income', money(income))
    c2.metric('Spent this month', money(spent), delta_color='inverse',
              delta=f'{money(spent - spent_last)} vs last month' if spent_last else None)
    c3.metric('Left this month', money(income - spent))
    c4.metric('Savings goal', money(goal), delta=f'{money(income - spent - goal)} vs goal')
    if income > 0:
        used = spent / allowance if allowance > 0 else 1.0
        st.progress(min(used, 1.0), text=f'{used:.0%} of your spending allowance used '
                                         f'({money(spent)} of {money(allowance)})')
    else:
        st.info('Add your monthly income in 👤 Profile to track your budget.')

    left, right = st.columns([1.1, 1])
    with left.container(border=True):
        st.subheader('Where your money went')
        if this_month.empty:
            st.info('No expenses this month yet. Add one from 🧾 Expenses.')
        else:
            donut_chart(this_month)
    with right.container(border=True):
        st.subheader('Recent transactions')
        if df.empty:
            st.info('Nothing recorded yet.')
        else:
            recent_transactions(df.head(6))

elif page == '🧾 Expenses':
    header('🧾 Expenses', 'Add new spending and manage what you have recorded')
    with st.container(border=True):
        st.subheader('Add an expense')
        # Category sits outside the form so the subcategory list updates straight away.
        c1, c2 = st.columns(2)
        category = c1.selectbox('Category', list(CATEGORIES))
        subcategory = c2.selectbox('Subcategory', CATEGORIES[category])
        with st.form('add_expense', clear_on_submit=True, border=False):
            c3, c4 = st.columns(2)
            expense_date = c3.date_input('Date', value=today, max_value=today, format='DD/MM/YYYY')
            amount = c4.number_input('Amount (₹)', min_value=0.0, step=50.0)
            note = st.text_input('Note (optional)', placeholder='e.g. Dinner with friends')
            saved = st.form_submit_button('➕ Add expense', type='primary')
        if saved:
            if amount <= 0:
                st.error('Amount must be greater than zero.')
            else:
                database.add_expense(DB, user['user_id'], expense_date.isoformat(),
                                     category, subcategory, amount, note.strip())
                flash(f'Added {money(amount)} to {category}.')
                st.rerun()

    with st.container(border=True):
        st.subheader('Your expenses')
        if df.empty:
            st.info('No expenses recorded yet.')
        else:
            f1, f2 = st.columns([2, 1])
            search = f1.text_input('Search', placeholder='🔍 Search notes or subcategories',
                                   label_visibility='collapsed')
            chosen = f2.multiselect('Categories', list(CATEGORIES), placeholder='All categories',
                                    label_visibility='collapsed')
            view = df
            if search:
                view = view[view['note'].str.contains(search, case=False, regex=False)
                            | view['subcategory'].str.contains(search, case=False, regex=False)]
            if chosen:
                view = view[view['category'].isin(chosen)]
            st.dataframe(view, hide_index=True,
                         column_order=['date', 'category', 'subcategory', 'amount', 'note'],
                         column_config={
                             'date': st.column_config.DateColumn('Date', format='DD MMM YYYY'),
                             'category': 'Category', 'subcategory': 'Subcategory',
                             'amount': st.column_config.NumberColumn('Amount', format='₹%.2f'),
                             'note': 'Note'})
            st.caption(f'{len(view)} expenses · total {money(view["amount"].sum())}')
            with st.expander('🗑️ Delete an expense'):
                labels = {r.id: f'{r.date:%d %b %Y} · {r.category} / {r.subcategory} · {money(r.amount)}'
                          for r in df.itertuples()}
                to_delete = st.selectbox('Choose the expense to delete', list(labels),
                                         format_func=labels.get)
                if st.button('Delete this expense'):
                    database.delete_expense(DB, user['user_id'], int(to_delete))
                    flash('Expense deleted.')
                    st.rerun()

elif page == '📊 Analytics':
    header('📊 Analytics', 'Spot patterns in your spending')
    if df.empty:
        st.info('Add some expenses to see analytics.')
    else:
        periods = {'This month': 1, 'Last 3 months': 3, 'Last 6 months': 6,
                   'Last 12 months': 12, 'All time': None}
        choice = st.radio('Period', list(periods), index=4, horizontal=True,
                          label_visibility='collapsed')
        months = periods[choice]
        data = df
        if months:
            start = (pd.Timestamp.today().to_period('M') - (months - 1)).to_timestamp()
            data = df[df['date'] >= start]

        if data.empty:
            st.info('No expenses in this period.')
        else:
            totals = data.groupby('category')['amount'].sum()
            a, b, c, d = st.columns(4)
            a.metric('Total spent', money(totals.sum()))
            b.metric('Transactions', len(data))
            c.metric('Average transaction', money(data['amount'].mean()))
            d.metric('Top category', totals.idxmax())
            left, right = st.columns(2)
            with left.container(border=True):
                st.subheader('Spending by category')
                donut_chart(data)
            with right.container(border=True):
                st.subheader('Top subcategories')
                subcategory_chart(data)
            with st.container(border=True):
                st.subheader('Monthly trend')
                monthly_chart(data)

elif page == '🤖 Prediction':
    header('🤖 Next-month prediction', 'A machine-learning estimate of what you will spend next month')
    model = get_model()
    if model is None:
        st.warning('The prediction model has not been trained yet.')
        if st.button('Train the model now', type='primary'):
            with st.spinner('Training on the sample dataset…'):
                result = subprocess.run([sys.executable, str(ROOT / 'train_model.py')],
                                        capture_output=True, text=True, cwd=ROOT)
            if result.returncode == 0:
                get_model.clear()
                flash('Model trained.')
                st.rerun()
            else:
                st.error('Training failed.')
                st.code(result.stderr)
    else:
        # Pre-fill with last month's real spending (or this month's, for new users).
        base, base_name = (last_month, 'last month') if not last_month.empty else (this_month, 'this month')
        by_category = base.groupby('category')['amount'].sum()
        if not base.empty:
            st.caption(f'Pre-filled from your {base_name} expenses. Change anything you like.')
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            income_in = c1.number_input('Monthly income (₹)', min_value=0.0, value=income, step=1000.0)
            age = c2.number_input('Age', min_value=18, max_value=100, value=25)
            goal_in = c3.number_input('Savings goal (₹)', min_value=0.0, value=goal, step=500.0)
            c4, c5, c6, c7 = st.columns(4)
            food = c4.number_input('Food (₹)', min_value=0.0, step=100.0,
                                   value=float(by_category.get('Food', 0)))
            travel = c5.number_input('Travel (₹)', min_value=0.0, step=100.0,
                                     value=float(by_category.get('Travel', 0)))
            shopping = c6.number_input('Shopping (₹)', min_value=0.0, step=100.0,
                                       value=float(by_category.get('Shopping', 0)))
            bills = c7.number_input('Bills (₹)', min_value=0.0, step=100.0,
                                    value=float(by_category.get('Bills', 0)))
            # In the training data, the previous month's total is exactly these four added up.
            previous = food + travel + shopping + bills
            st.caption(f'Month total used by the model: **{money(previous)}** '
                       '(food + travel + shopping + bills)')
            predict = st.button('🔮 Predict next month', type='primary')

        if predict:
            values = {'monthly_income': income_in, 'age': age, 'savings_goal': goal_in,
                      'food_expense': food, 'travel_expense': travel,
                      'shopping_expense': shopping, 'bills_expense': bills,
                      'previous_month_expense': previous}
            predicted = predict_expense(model, values)
            database.add_prediction(DB, user['user_id'], predicted)
            allowance = income_in - goal_in
            r1, r2, r3 = st.columns(3)
            r1.metric('Predicted next month', money(predicted), delta_color='inverse',
                      delta=f'{money(predicted - previous)} vs this input')
            r2.metric('Spending allowance', money(allowance), help='Income minus savings goal')
            r3.metric('Model accuracy (R²)', f"{model['r2_score']:.2f}",
                      help=f"{model.get('model_name', 'Model')}, typical error ±{money(model['mae'])}")
            if predicted > allowance:
                st.warning(f'At this rate you would miss your savings goal by {money(predicted - allowance)}. '
                           'See 💡 Budget Advisor for ideas.')
            else:
                st.success(f'Looks good. You would still have {money(allowance - predicted)} spare '
                           'after your savings goal.')
            compare = pd.DataFrame({'label': ['Input month', 'Predicted next month', 'Allowance'],
                                    'amount': [previous, predicted, allowance]})
            chart = alt.Chart(compare).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X('amount:Q', title='Amount (₹)'),
                y=alt.Y('label:N', sort=None, title=None),
                color=alt.Color('label:N', legend=None,
                                scale=alt.Scale(range=['#94A3B8', '#4F46E5', '#10B981'])),
                tooltip=['label:N', alt.Tooltip('amount:Q', title='Amount (₹)', format=',.0f')])
            st.altair_chart(chart.properties(height=170))

elif page == '💡 Budget Advisor':
    header('💡 Budget Advisor', f'How {today:%B} is going against your plan')
    if income <= 0:
        st.warning('Add your monthly income in 👤 Profile to get budget advice.')
    else:
        spent = this_month['amount'].sum()
        allowance = income - goal
        left_to_spend = allowance - spent
        days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
        a, b, c, d = st.columns(4)
        a.metric('Income', money(income))
        b.metric('Savings goal', money(goal))
        c.metric('Spent this month', money(spent))
        d.metric('Safe to spend', money(max(left_to_spend, 0)), help='Income − savings goal − spent so far')
        used = spent / allowance if allowance > 0 else 1.0
        st.progress(min(used, 1.0), text=f'{used:.0%} of your spending allowance used')
        if spent > income:
            st.error(f'🚨 Overspending: you have spent {money(spent - income)} more than your income this month.')
        elif left_to_spend < 0:
            st.warning(f'⚠️ You are {money(-left_to_spend)} over your spending allowance, '
                       "so this month's savings goal is at risk.")
        else:
            st.success(f'✅ You can spend about **{money(left_to_spend / days_left)} a day** for the next '
                       f'{days_left} days and still save {money(goal)}.')

        if not this_month.empty:
            totals = this_month.groupby('category')['amount'].sum().sort_values(ascending=False)
            top, top_amount = totals.index[0], totals.iloc[0]
            table = totals.reset_index()
            table['share'] = table['amount'] / totals.sum() * 100
            left, right = st.columns(2)
            with left.container(border=True):
                st.subheader('This month by category')
                st.dataframe(table, hide_index=True, column_config={
                    'category': 'Category',
                    'amount': st.column_config.NumberColumn('Amount', format='₹%.0f'),
                    'share': st.column_config.ProgressColumn('Share', format='%.0f%%',
                                                             min_value=0, max_value=100)})
            with right.container(border=True):
                st.subheader('Tip for you')
                st.markdown(f'Your biggest category this month is **{top}**: {money(top_amount)}, '
                            f'or {top_amount / totals.sum():.0%} of your spending.')
                st.info(TIPS.get(top, TIPS['Other']))

elif page == '🗄️ Reports':
    header('🗄️ Reports', 'Download your data and review past predictions')
    if df.empty:
        st.info('No saved expenses yet.')
    else:
        with st.container(border=True):
            st.subheader('Monthly summary')
            summary = df.assign(Month=df['date'].dt.strftime('%Y-%m')).pivot_table(
                index='Month', columns='category', values='amount', aggfunc='sum', fill_value=0)
            summary['Total'] = summary.sum(axis=1)
            summary = summary.sort_index(ascending=False)
            st.dataframe(summary, column_config={
                col: st.column_config.NumberColumn(format='₹%.0f') for col in summary.columns})
            st.download_button('⬇️ Download summary (CSV)', summary.to_csv().encode('utf-8'),
                               'monthly_summary.csv', 'text/csv')
        with st.container(border=True):
            st.subheader('All transactions')
            export = df.drop(columns='id').assign(date=df['date'].dt.date)
            st.dataframe(export, hide_index=True, column_config={
                'amount': st.column_config.NumberColumn('amount', format='₹%.2f')})
            st.download_button('⬇️ Download transactions (CSV)', export.to_csv(index=False).encode('utf-8'),
                               'my_expenses.csv', 'text/csv')
    with st.container(border=True):
        st.subheader('Saved predictions')
        predictions = database.get_predictions(DB, user['user_id'])
        if predictions:
            pred_df = pd.DataFrame(predictions, columns=['ID', 'Predicted amount', 'Saved on (UTC)'])
            st.dataframe(pred_df.drop(columns='ID'), hide_index=True, column_config={
                'Predicted amount': st.column_config.NumberColumn(format='₹%.0f')})
        else:
            st.info('No predictions saved yet. Run one from 🤖 Prediction.')

elif page == '👤 Profile':
    header('👤 Profile', 'Update your details and keep your account secure')
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader('Your details')
        with st.form('profile', border=False):
            name = st.text_input('Full name', value=user['name'])
            st.text_input('Phone number (used to log in)', value=user['phone'], disabled=True)
            new_income = st.number_input('Monthly income (₹)', min_value=0.0, step=1000.0, value=income)
            new_goal = st.number_input('Monthly savings goal (₹)', min_value=0.0, step=500.0, value=goal)
            save = st.form_submit_button('Save changes', type='primary')
        if save:
            if not name.strip():
                st.error('Name cannot be empty.')
            else:
                database.update_profile(DB, user['user_id'], name.strip(), new_income, new_goal)
                set_current_user(database.get_user(DB, user['user_id']))
                flash('Profile updated.')
                st.rerun()
    with right.container(border=True):
        st.subheader('Change password')
        with st.form('password', clear_on_submit=True, border=False):
            current = st.text_input('Current password', type='password')
            new = st.text_input('New password', type='password', help='At least 8 characters')
            confirm = st.text_input('Confirm new password', type='password')
            change = st.form_submit_button('Update password')
        if change:
            stored = database.get_user(DB, user['user_id'])['password_hash']
            if not auth.verify_password(current, stored):
                st.error('Current password is incorrect.')
            elif len(new) < 8:
                st.error('New password must be at least 8 characters.')
            elif new != confirm:
                st.error('New passwords do not match.')
            else:
                database.update_password(DB, user['user_id'], auth.hash_password(new))
                st.success('Password updated.')

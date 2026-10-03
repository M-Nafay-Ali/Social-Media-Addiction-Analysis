import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Page Configuration
st.set_page_config(
    page_title="Social Media Addiction Risk Analysis",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Behavioral & Psychological Drivers of Social Media Addiction")
st.markdown("""
This application analyzes behavioral patterns, psychological triggers, and screen usage to evaluate **Social Media Addiction Risk Scores**.
""")

# Load Dataset
@st.cache_data
def load_data():
    # Update path if hosting locally vs relative path
    df = pd.read_csv("data/social_media_dopamine_productivity.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Please ensure the dataset is present in 'data/social_media_dopamine_productivity.csv'. Error: {e}")
    st.stop()

# Sidebar - Options
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "Feature Importance & Correlation", "Live Risk Predictor"])

# Define Features
numeric_features = [
    'avg_daily_sm_hours', 'dopamine_rush_feel_score', 
    'reward_seeking_behavior', 'fomo_score', 'comparison_to_others_score'
]
categorical_features = [
    'boredom_to_phone_reflex', 'inability_to_delay_gratification',
    'late_night_scrolling', 'first_check_of_day'
]

# Preprocessing & Model Pipeline setup
X = df[numeric_features + categorical_features]
y = df['sm_addiction_risk_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

model = Ridge()
model.fit(X_train_preprocessed, y_train)

y_pred = model.predict(X_test_preprocessed)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Page 1: Executive Summary
if page == "Executive Summary":
    st.header("📌 Executive Summary")
    st.markdown("""
    - **Objective:** Investigate and rank core behavioral and psychological factors associated with social media addiction risk.
    - **Methodology:** Leakage-free preprocessing pipeline utilizing `ColumnTransformer`, `SelectKBest`, and `Ridge Regression`.
    - **Model Performance:** 
    """)
    
    col1, col2 = st.columns(2)
    col1.metric(label="Test R² Score", value=f"{r2:.4f}")
    col2.metric(label="Test RMSE", value=f"{rmse:.4f}")

    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))

# Page 2: Feature Importance & Correlation
elif page == "Feature Importance & Correlation":
    st.header("📊 Feature Importance Analysis")

    encoded_cat_names = preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(encoded_cat_names)

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=coef_df.head(10), 
        x='Coefficient', 
        y='Feature', 
        hue='Feature', 
        palette='viridis', 
        legend=False,
        ax=ax
    )
    ax.set_title('Top 10 Ridge Regression Feature Coefficients')
    st.pyplot(fig)

# Page 3: Live Risk Predictor
elif page == "Live Risk Predictor":
    st.header("🔮 Estimate Addiction Risk Score")
    st.write("Adjust the behavioral levers below to generate a model prediction:")

    col1, col2 = st.columns(2)

    with col1:
        avg_sm_hours = st.slider("Average Daily Social Media Hours", 0.0, 15.0, 5.0)
        dopamine_score = st.slider("Dopamine Rush Feel Score (1-10)", 1, 10, 6)
        fomo_score = st.slider("FOMO Score (1-10)", 1, 10, 5)
        reward_seeking = st.slider("Reward Seeking Behavior (1-10)", 1, 10, 5)
        comparison_score = st.slider("Comparison to Others Score (1-10)", 1, 10, 5)

    with col2:
        boredom_reflex = st.selectbox("Boredom to Phone Reflex", df['boredom_to_phone_reflex'].dropna().unique())
        delay_grat = st.selectbox("Inability to Delay Gratification", df['inability_to_delay_gratification'].dropna().unique())
        late_night = st.selectbox("Late Night Scrolling Habits", df['late_night_scrolling'].dropna().unique())
        first_check = st.selectbox("First Check of the Day", df['first_check_of_day'].dropna().unique())

    # Build input dataframe
    user_input = pd.DataFrame([{
        'avg_daily_sm_hours': avg_sm_hours,
        'dopamine_rush_feel_score': dopamine_score,
        'reward_seeking_behavior': reward_seeking,
        'fomo_score': fomo_score,
        'comparison_to_others_score': comparison_score,
        'boredom_to_phone_reflex': boredom_reflex,
        'inability_to_delay_gratification': delay_grat,
        'late_night_scrolling': late_night,
        'first_check_of_day': first_check
    }])

    user_input_preprocessed = preprocessor.transform(user_input)
    prediction = model.predict(user_input_preprocessed)[0]

    st.markdown("---")
    st.subheader(f"Predicted Addiction Risk Score: **{prediction:.2f} / 10**")

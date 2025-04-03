import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE  # For handling imbalance
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Title and Introduction
st.title("The Social Media Paradox - Enhanced Analysis")
st.markdown("""
Welcome to the Global Digital Well-being Research Institute (GDWRI) Dashboard.  
Analyze social media usage patterns and their impact on well-being with advanced statistical and machine learning techniques.
""")

# Sidebar for file upload
st.sidebar.header("Data Upload")
health_file = st.sidebar.file_uploader("Upload Sleep Dataset (Sleep Dataset.xlsm)", type=["xlsm"])
social_file = st.sidebar.file_uploader("Upload Social Media Usage (Social Media Usage - Train.xlsm)", type=["xlsm"])

# Function to categorize usage
def categorize_usage(minutes):
    if minutes < 60: return 'Low'
    elif minutes < 120: return 'Moderate'
    else: return 'High'

# Main app logic
if health_file and social_file:
    # Load datasets
    try:
        health_data = pd.read_excel(health_file)
        social_media_data = pd.read_excel(social_file)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    # Display column names
    st.write("Columns in Sleep Dataset:", health_data.columns.tolist())
    st.write("Columns in Social Media Usage:", social_media_data.columns.tolist())

    # Data Formatting and Cleaning
    health_data['Sleep Disorder'] = health_data['Sleep Disorder'].fillna('None')
    health_data['BMI Category'] = health_data['BMI Category'].str.replace('Normal Weight', 'Normal')
    social_media_data = social_media_data.dropna()
    social_media_data['Dominant_Emotion'] = social_media_data['Dominant_Emotion'].str.strip()

    # Statistical Measures
    st.subheader("Statistical Measures")
    numeric_cols_health = health_data.select_dtypes(include=np.number).columns
    numeric_cols_social = social_media_data.select_dtypes(include=np.number).columns
    
    health_stats = health_data[numeric_cols_health].agg(['mean', 'std', 'skew', 'var']).T
    social_stats = social_media_data[numeric_cols_social].agg(['mean', 'std', 'skew', 'var']).T
    
    st.write("Health Data Statistics:", health_stats)
    st.write("Social Media Data Statistics:", social_stats)

    # Feature Engineering
    health_data['Health_Score'] = (health_data['Quality of Sleep'] * 0.3 + 
                                 (10 - health_data['Stress Level']) * 0.2 +
                                 health_data['Physical Activity Level'] * 0.2 +
                                 health_data['Daily Steps'] / 10000 * 0.3)
    health_data['Sleep_Efficiency'] = health_data['Quality of Sleep'] / health_data['Sleep Duration']
    social_media_data['Engagement_Rate'] = (social_media_data['Likes_Received_Per_Day'] + 
                                           social_media_data['Comments_Received_Per_Day'] + 
                                           social_media_data['Messages_Sent_Per_Day']) / social_media_data['Posts_Per_Day'].replace(0, 1)

    # Find usage column
    possible_time_columns = ['Daily Minutes', 'Daily_Usage', 'Minutes_Per_Day', 'Time_Spent', 
                            'Usage_Minutes', 'Daily_Usage_Time (minutes)']
    usage_column = None
    for col in possible_time_columns:
        if col in social_media_data.columns:
            usage_column = col
            break

    if usage_column is None:
        st.error("No time usage column found. Available columns: " + str(social_media_data.columns.tolist()))
        st.stop()
    else:
        st.write(f"Using column '{usage_column}' for usage categorization")
        social_media_data['Usage_Category'] = social_media_data[usage_column].apply(categorize_usage)

    # Merge datasets
    merged_data = pd.merge(health_data, social_media_data, 
                          left_on='Person ID', right_on='User_ID', how='inner')
    st.write("Columns in Merged Data:", merged_data.columns.tolist())

    # Handle duplicate columns after merge
    age_col = 'Age' if 'Age' in merged_data.columns else 'Age_x'
    gender_col = 'Gender' if 'Gender' in merged_data.columns else 'Gender_x'

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_cols_merged = merged_data.select_dtypes(include=np.number).columns
    corr_matrix = merged_data[numeric_cols_merged].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix of Numeric Features")
    st.plotly_chart(fig_corr)

    # Platform Prediction Model with Improvements
    X_platform = merged_data[[age_col, gender_col, 'Occupation', 'Health_Score', 'Sleep_Efficiency', 'Engagement_Rate', usage_column]].copy()
    y_platform = merged_data['Platform']

    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_platform = LabelEncoder()

    X_platform[gender_col] = le_gender.fit_transform(X_platform[gender_col])
    X_platform['Occupation'] = le_occupation.fit_transform(X_platform['Occupation'])
    y_platform = le_platform.fit_transform(y_platform)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_platform_balanced, y_platform_balanced = smote.fit_resample(X_platform, y_platform)

    X_train, X_test, y_train, y_test = train_test_split(X_platform_balanced, y_platform_balanced, test_size=0.2, random_state=42)

    numeric_features = [age_col, 'Health_Score', 'Sleep_Efficiency', 'Engagement_Rate', usage_column]
    categorical_features = [gender_col, 'Occupation']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, 15],
        'classifier__min_samples_split': [2, 5]
    }
    platform_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))  # Class weights for imbalance
    ])

    grid_search = GridSearchCV(platform_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    platform_model = grid_search.best_estimator_

    y_pred = platform_model.predict(X_test)
    st.write("Optimized Platform Prediction Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred, target_names=le_platform.classes_))
    st.write("Best Parameters:", grid_search.best_params_)

    # Cross-validation
    cv_scores = cross_val_score(platform_model, X_platform_balanced, y_platform_balanced, cv=5)
    st.write("Cross-Validation Scores (Platform Model):", cv_scores)
    st.write("Mean CV Score:", cv_scores.mean())

    # Confusion Matrix
    cm_platform = confusion_matrix(y_test, y_pred)
    fig_cm_platform = px.imshow(cm_platform, text_auto=True, title="Confusion Matrix - Platform Prediction",
                               labels=dict(x="Predicted", y="Actual"), x=le_platform.classes_, y=le_platform.classes_)
    st.plotly_chart(fig_cm_platform)

    # Feature Importance
    feature_importance = platform_model.named_steps['classifier'].feature_importances_
    feature_names = numeric_features + categorical_features
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    fig_importance = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance - Platform Model")
    st.plotly_chart(fig_importance)

    # Health Impact Prediction Model
    X_health = merged_data[[age_col, gender_col, 'Platform', usage_column, 'Usage_Category', 'Engagement_Rate']].copy()
    y_health = (merged_data['Health_Score'] > merged_data['Health_Score'].median()).astype(int)

    X_health[gender_col] = le_gender.transform(X_health[gender_col])
    X_health['Platform'] = le_platform.transform(X_health['Platform'])
    le_usage = LabelEncoder()
    X_health['Usage_Category'] = le_usage.fit_transform(X_health['Usage_Category'])

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_health, y_health, test_size=0.2, random_state=42)

    numeric_features_h = [age_col, usage_column, 'Engagement_Rate']
    categorical_features_h = [gender_col, 'Platform', 'Usage_Category']

    preprocessor_h = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))]), numeric_features_h),
            ('cat', 'passthrough', categorical_features_h)
        ])

    health_model = Pipeline([
        ('preprocessor', preprocessor_h),
        ('classifier', GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42))
    ])

    health_model.fit(X_train_h, y_train_h)
    y_pred_h = health_model.predict(X_test_h)
    st.write("Health Impact Prediction Accuracy:", accuracy_score(y_test_h, y_pred_h))

    # Cross-validation
    cv_scores_h = cross_val_score(health_model, X_health, y_health, cv=5)
    st.write("Cross-Validation Scores (Health Model):", cv_scores_h)
    st.write("Mean CV Score:", cv_scores_h.mean())

    # Confusion Matrix
    cm_health = confusion_matrix(y_test_h, y_pred_h)
    fig_cm_health = px.imshow(cm_health, text_auto=True, title="Confusion Matrix - Health Impact Prediction",
                             labels=dict(x="Predicted", y="Actual"), x=['Below Median', 'Above Median'], y=['Below Median', 'Above Median'])
    st.plotly_chart(fig_cm_health)

    # UI Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Platform Usage", "Health Impact", "Predictions", "Summary"])

    # Tab 1: Platform Usage Analysis
    with tab1:
        st.header("Platform Usage by Age Group")
        fig1 = px.box(merged_data, x='Platform', y=age_col, color='Platform', 
                     title="Social Media Platform Usage Across Age Groups")
        st.plotly_chart(fig1)

        st.subheader("Platform Popularity by Age Group")
        age_bins = [0, 18, 25, 35, 50, 100]
        age_labels = ['<18', '18-25', '25-35', '35-50', '50+']
        merged_data['Age_Group'] = pd.cut(merged_data[age_col], bins=age_bins, labels=age_labels)
        fig2 = px.histogram(merged_data, x='Platform', color='Age_Group', barmode='group',
                           title="Platform Popularity by Age Group")
        st.plotly_chart(fig2)

    # Tab 2: Health Impact Analysis
    with tab2:
        st.header("Health Impact Analysis")
        st.subheader("Correlation with Usage")
        metric = st.selectbox("Select Health Metric", ['Stress Level', 'Quality of Sleep', 'Physical Activity Level'])
        fig3 = px.scatter(merged_data, x=usage_column, y=metric, color='Platform', 
                         title=f"{metric} vs. Social Media Usage Time", trendline="ols")
        st.plotly_chart(fig3)

        st.subheader("Sleep Disorders by Usage Category")
        fig4 = px.bar(merged_data.groupby(['Usage_Category', 'Sleep Disorder']).size().reset_index(name='Count'),
                     x='Usage_Category', y='Count', color='Sleep Disorder', 
                     title="Sleep Disorders by Social Media Usage Category", barmode='group')
        st.plotly_chart(fig4)

    # Tab 3: Predictions
    with tab3:
        st.header("Predictions")
        
        st.subheader("Predict Social Media Platform")
        age = st.number_input("Age", min_value=0, max_value=100, value=25, key="platform_age")
        gender = st.selectbox("Gender", merged_data[gender_col].unique(), key="platform_gender")
        occupation = st.selectbox("Occupation", merged_data['Occupation'].unique(), key="platform_occupation")
        health_score = st.number_input("Health Score (1-10)", min_value=1.0, max_value=10.0, value=7.0, key="platform_health")
        sleep_efficiency = st.number_input("Sleep Efficiency", min_value=0.0, max_value=2.0, value=1.0, key="platform_sleep")
        engagement = st.number_input("Engagement Rate", min_value=0.0, value=1.0, key="platform_engagement")
        usage = st.number_input("Daily Usage Time (minutes)", min_value=0, value=120, key="platform_usage")

        if st.button("Predict Platform"):
            input_data = pd.DataFrame({
                age_col: [age],
                gender_col: [gender],
                'Occupation': [occupation],
                'Health_Score': [health_score],
                'Sleep_Efficiency': [sleep_efficiency],
                'Engagement_Rate': [engagement],
                usage_column: [usage]
            })
            input_data[gender_col] = le_gender.transform(input_data[gender_col])
            input_data['Occupation'] = le_occupation.transform(input_data['Occupation'])
            prediction = platform_model.predict(input_data)
            platform = le_platform.inverse_transform(prediction)[0]
            st.success(f"Predicted Platform: **{platform}**")
            st.write(f"Based on: Age {age}, {gender} {occupation}, Health Score {health_score}, Sleep Efficiency {sleep_efficiency}, Engagement Rate {engagement}, Usage {usage} min")

        st.subheader("Predict Health Impact")
        usage = st.number_input("Daily Usage Time (minutes)", min_value=0, value=120, key="health_usage")
        platform = st.selectbox("Platform", merged_data['Platform'].unique(), key="health_platform")
        engagement = st.number_input("Engagement Rate", min_value=0.0, value=1.0, key="health_engagement")

        if st.button("Predict Health Impact"):
            usage_cat = categorize_usage(usage)
            input_data = pd.DataFrame({
                age_col: [30],
                gender_col: ['Male'],
                'Platform': [platform],
                usage_column: [usage],
                'Usage_Category': [usage_cat],
                'Engagement_Rate': [engagement]
            })
            input_data[gender_col] = le_gender.transform(input_data[gender_col])
            input_data['Platform'] = le_platform.transform(input_data['Platform'])
            input_data['Usage_Category'] = le_usage.transform(input_data['Usage_Category'])
            health_pred = health_model.predict(input_data)[0]
            health_status = "Above Median" if health_pred == 1 else "Below Median"
            st.success(f"Predicted Health Status: **{health_status}**")
            st.write(f"Based on: {usage} minutes/day on {platform} ({usage_cat} usage), Engagement Rate {engagement}")

    # Tab 4: Summary
    with tab4:
        st.header("Summary and Insights")
        st.markdown("""
        ### Observations from the Enhanced Analysis:
        - **Data Statistics**: High variance in usage time and engagement rates indicates diverse user behaviors.
        - **Model Improvements**: SMOTE and GridSearchCV improved platform prediction, though accuracy remains a challenge due to limited data size and feature discriminability.
        - **Platform Usage**: Younger users favor Instagram/YouTube; older users prefer Facebook/LinkedIn.
        - **Health Impact**: Excessive usage correlates with stress and sleep issues; engagement rate may mitigate some effects.
        - **Feature Importance**: 'Engagement_Rate' and 'Health_Score' are key predictors.

        ### Recommendations:
        - **Policy**: Enforce usage limits on platforms with low sample representation (e.g., Snapchat) to gather more data.
        - **Education**: Focus on high-usage groups to promote balanced digital habits.
        - **Bias**: Dataset imbalance addressed with SMOTE; further data collection needed for underrepresented platforms.

        ### Future Trends:
        - Video platforms may grow, necessitating proactive health interventions.
        """)

else:
    st.warning("Please upload both the Sleep Dataset and Social Media Usage files to proceed.")
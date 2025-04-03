import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import joblib

# Load datasets
try:
    health_data = pd.read_excel('Sleep Dataset.xlsm')
    social_media_data = pd.read_excel('Social Media Usage - Train.xlsm')
except FileNotFoundError:
    print("Error: One or both Excel files not found. Please check file names and paths.")
    exit()

# Print available columns for debugging
print("Columns in social_media_data:", social_media_data.columns.tolist())
print("Columns in health_data:", health_data.columns.tolist())

# Data Preprocessing
health_data['Sleep Disorder'] = health_data['Sleep Disorder'].fillna('None')
health_data['BMI Category'] = health_data['BMI Category'].str.replace('Normal Weight', 'Normal')
social_media_data = social_media_data.dropna()
social_media_data['Dominant_Emotion'] = social_media_data['Dominant_Emotion'].str.strip()

# Feature Engineering
health_data['Health_Score'] = (health_data['Quality of Sleep'] * 0.3 + 
                             (10 - health_data['Stress Level']) * 0.2 +
                             health_data['Physical Activity Level'] * 0.2 +
                             health_data['Daily Steps'] / 10000 * 0.3)

def categorize_usage(minutes):
    if minutes < 60: return 'Low'
    elif minutes < 120: return 'Moderate'
    else: return 'High'

# Find appropriate usage time column
possible_time_columns = ['Daily Minutes', 'Daily_Usage', 'Minutes_Per_Day', 'Time_Spent', 
                        'Usage_Minutes', 'Daily_Usage_Time (minutes)']  # Explicitly includes your column
usage_column = None
for col in possible_time_columns:
    if col in social_media_data.columns:
        usage_column = col
        break

if usage_column is None:
    raise ValueError("No time usage column found. Available columns: " + str(social_media_data.columns.tolist()))
else:
    print(f"Using column '{usage_column}' for usage categorization")
    social_media_data['Usage_Category'] = social_media_data[usage_column].apply(categorize_usage)

# Merge datasets
merged_data = pd.merge(health_data, social_media_data, 
                      left_on='Person ID', right_on='User_ID', how='inner')

# Platform Prediction Model
X_platform = merged_data[['Age', 'Gender', 'Occupation', 'Health_Score']].copy()
y_platform = merged_data['Platform']

le_gender = LabelEncoder()
le_occupation = LabelEncoder()
le_platform = LabelEncoder()

X_platform['Gender'] = le_gender.fit_transform(X_platform['Gender'])
X_platform['Occupation'] = le_occupation.fit_transform(X_platform['Occupation'])
y_platform = le_platform.fit_transform(y_platform)

X_train, X_test, y_train, y_test = train_test_split(X_platform, y_platform, test_size=0.2, random_state=42)

numeric_features = ['Age', 'Health_Score']
categorical_features = ['Gender', 'Occupation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ])

platform_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

platform_model.fit(X_train, y_train)

y_pred = platform_model.predict(X_test)
print("Platform Prediction Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_platform.classes_))

joblib.dump(platform_model, 'platform_predictor.joblib')
joblib.dump(le_platform, 'platform_encoder.joblib')

# Health Impact Prediction Model
X_health = merged_data[['Age', 'Gender', 'Platform', usage_column, 'Usage_Category']].copy()
y_health = (merged_data['Health_Score'] > merged_data['Health_Score'].median()).astype(int)

X_health['Gender'] = le_gender.transform(X_health['Gender'])
X_health['Platform'] = le_platform.transform(X_health['Platform'])
le_usage = LabelEncoder()
X_health['Usage_Category'] = le_usage.fit_transform(X_health['Usage_Category'])

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_health, y_health, test_size=0.2, random_state=42)

numeric_features_h = ['Age', usage_column]
categorical_features_h = ['Gender', 'Platform', 'Usage_Category']

preprocessor_h = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_h),
        ('cat', 'passthrough', categorical_features_h)
    ])

health_model = Pipeline([
    ('preprocessor', preprocessor_h),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

health_model.fit(X_train_h, y_train_h)

y_pred_h = health_model.predict(X_test_h)
print("\nHealth Impact Prediction Accuracy:", accuracy_score(y_test_h, y_pred_h))

joblib.dump(health_model, 'health_predictor.joblib')
joblib.dump(le_usage, 'usage_encoder.joblib')

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Social Media Well-being Dashboard"),
    
    dcc.Tabs([
        dcc.Tab(label='Platform Analysis', children=[
            html.H3("Platform Usage by Age Group"),
            dcc.Graph(id='platform-age-plot'),
            
            html.H3("Platform Popularity by Occupation"),
            dcc.Dropdown(
                id='occupation-dropdown',
                options=[{'label': occ, 'value': occ} for occ in merged_data['Occupation'].unique()],
                value='Software Engineer'
            ),
            dcc.Graph(id='platform-occupation-plot')
        ]),
        
        dcc.Tab(label='Health Impact', children=[
            html.H3("Health Metrics by Platform Usage"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Stress Level', 'value': 'Stress Level'},
                    {'label': 'Sleep Quality', 'value': 'Quality of Sleep'},
                    {'label': 'Physical Activity', 'value': 'Physical Activity Level'}
                ],
                value='Stress Level'
            ),
            dcc.Graph(id='health-metric-plot'),
            
            html.H3("Sleep Disorders by Usage Category"),
            dcc.Graph(id='sleep-disorder-plot')
        ]),
        
        dcc.Tab(label='Predictions', children=[
            html.H3("Predict Social Media Platform"),
            html.Div([
                html.Label("Age:"),
                dcc.Input(id='pred-age', type='number', value=25),
                
                html.Label("Gender:"),
                dcc.Dropdown(
                    id='pred-gender',
                    options=[{'label': g, 'value': g} for g in merged_data['Gender'].unique()],
                    value='Male'
                ),
                
                html.Label("Occupation:"),
                dcc.Dropdown(
                    id='pred-occupation',
                    options=[{'label': occ, 'value': occ} for occ in merged_data['Occupation'].unique()],
                    value='Software Engineer'
                ),
                
                html.Label("Health Score (1-10):"),
                dcc.Input(id='pred-health', type='number', value=7),
                
                html.Button('Predict Platform', id='predict-button')
            ]),
            
            html.Div(id='platform-prediction-output'),
            
            html.Hr(),
            
            html.H3("Predict Health Impact"),
            html.Div([
                html.Label("Daily Usage Time (minutes):"),
                dcc.Input(id='pred-usage', type='number', value=120),
                
                html.Label("Platform:"),
                dcc.Dropdown(
                    id='pred-platform',
                    options=[{'label': p, 'value': p} for p in merged_data['Platform'].unique()],
                    value='Instagram'
                ),
                
                html.Button('Predict Health Impact', id='predict-health-button')
            ]),
            
            html.Div(id='health-prediction-output')
        ])
    ])
])

# Callbacks
@app.callback(
    Output('platform-age-plot', 'figure'),
    Input('platform-age-plot', 'id')
)
def update_platform_age_plot(_):
    fig = px.box(merged_data, x='Platform', y='Age', 
                 title="Platform Usage by Age Group",
                 color='Platform')
    return fig

@app.callback(
    Output('platform-occupation-plot', 'figure'),
    Input('occupation-dropdown', 'value')
)
def update_platform_occupation_plot(occupation):
    filtered_data = merged_data[merged_data['Occupation'] == occupation]
    fig = px.pie(filtered_data, names='Platform', 
                 title=f"Platform Popularity among {occupation}s")
    return fig

@app.callback(
    Output('health-metric-plot', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_health_metric_plot(metric):
    fig = px.violin(merged_data, x='Usage_Category', y=metric, color='Platform',
                   title=f"{metric} by Social Media Usage Category")
    return fig

@app.callback(
    Output('sleep-disorder-plot', 'figure'),
    Input('sleep-disorder-plot', 'id')
)
def update_sleep_disorder_plot(_):
    disorder_counts = merged_data.groupby(['Usage_Category', 'Sleep Disorder']).size().reset_index(name='Count')
    fig = px.bar(disorder_counts, x='Usage_Category', y='Count', color='Sleep Disorder',
                title="Sleep Disorders by Social Media Usage Category",
                barmode='group')
    return fig

@app.callback(
    Output('platform-prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [Input('pred-age', 'value'),
     Input('pred-gender', 'value'),
     Input('pred-occupation', 'value'),
     Input('pred-health', 'value')]
)
def predict_platform(n_clicks, age, gender, occupation, health_score):
    if n_clicks is None:
        return ""
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Occupation': [occupation],
        'Health_Score': [health_score]
    })
    
    input_data['Gender'] = le_gender.transform(input_data['Gender'])
    input_data['Occupation'] = le_occupation.transform(input_data['Occupation'])
    
    prediction = platform_model.predict(input_data)
    platform = le_platform.inverse_transform(prediction)[0]
    
    return html.Div([
        html.H4("Predicted Platform:"),
        html.P(platform),
        html.P(f"Based on: Age {age}, {gender} {occupation} with health score {health_score}")
    ])

@app.callback(
    Output('health-prediction-output', 'children'),
    Input('predict-health-button', 'n_clicks'),
    [Input('pred-usage', 'value'),
     Input('pred-platform', 'value')]
)
def predict_health_impact(n_clicks, usage, platform):
    if n_clicks is None:
        return ""
    
    usage_cat = categorize_usage(usage)
    input_data = pd.DataFrame({
        'Age': [30],
        'Gender': ['Male'],
        'Platform': [platform],
        usage_column: [usage],
        'Usage_Category': [usage_cat]
    })
    
    input_data['Gender'] = le_gender.transform(input_data['Gender'])
    input_data['Platform'] = le_platform.transform(input_data['Platform'])
    input_data['Usage_Category'] = le_usage.transform(input_data['Usage_Category'])
    
    health_pred = health_model.predict(input_data)[0]
    health_status = "Above Median" if health_pred == 1 else "Below Median"
    
    return html.Div([
        html.H4("Predicted Health Impact:"),
        html.P(f"Health Status: {health_status}"),
        html.P(f"Based on: {usage} minutes/day on {platform} ({usage_cat} usage)")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
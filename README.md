# 🌟 The Social Media Paradox - Enhanced Analysis 🌟

Welcome to the **Global Digital Well-being Research Institute (GDWRI)** project! This repository contains a Streamlit-based dashboard and machine learning models to analyze social media usage patterns and their impact on well-being. Built with 💻 Python, 📊 machine learning, and ✨ interactive visualizations, this project aims to uncover insights into digital habits and health outcomes.

---

## 📂 Directory Structure

Here's a quick overview of what's inside:
tanish-24-git-cummins-.git/
├── app.py                  # 🚀 Main Streamlit app script
├── gender_encoder.joblib   # 🧑‍🤝‍🧑 Label encoder for gender
├── health_predictor.joblib # 🩺 Trained health impact prediction model
├── occupation_encoder.joblib # 💼 Label encoder for occupation
├── om.ipynb               # 📓 Jupyter notebook with initial analysis
├── platform_encoder.joblib # 🌐 Label encoder for platforms
├── platform_predictor.joblib # 📱 Trained platform prediction model
├── Sleep Dataset.xlsm     # 😴 Sleep and health dataset
├── Social Media Usage - Train.xlsm # 📲 Social media usage dataset
└── usage_encoder.joblib   # ⏱️ Label encoder for usage categories


---

## 📜 Project Overview

This project combines two datasets—**Sleep Dataset** and **Social Media Usage**—to:
- 🔍 Analyze how social media usage affects sleep, stress, and overall health.
- 🤖 Predict preferred social media platforms and health impacts using machine learning.
- 📈 Visualize trends with interactive charts powered by Plotly.

### Key Features
- **Data Cleaning**: Handles missing values and standardizes categories.
- **Feature Engineering**: Creates `Health_Score`, `Sleep_Efficiency`, and `Engagement_Rate`.
- **Models**: Uses RandomForest (platform prediction) and GradientBoosting (health impact) with SMOTE and GridSearchCV for optimization.
- **UI**: Streamlit dashboard with tabs for usage, health, predictions, and insights.

---

## 🛠️ Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/tanish-24-git-cummins/your-repo-name.git
   cd tanish-24-git-cummins-.git

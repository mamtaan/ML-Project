### End To End Machine Learning Project
# Customer Churn Prediction — End-to-End ML Project

This project predicts whether a telecom customer is likely to churn, based on their
account, service, and billing details.

The main goal here wasn't to squeeze out the highest possible accuracy — it was to build
a complete, working ML pipeline: raw data → cleaning → EDA → feature engineering →
model training → a deployed Flask app where someone can actually plug in customer
details and get a prediction back. So the F1 score below is decent for an imbalanced
dataset, but chasing a better number wasn't really the point of doing this.

## Overview

- **Dataset:** Telco Customer Churn (7,043 customers, 19 features + target)
- **Problem type:** Binary classification (Churn: Yes/No)
- **Best model:** Gradient Boosting (tuned via RandomizedSearchCV)
- **Final F1 Score:** 0.5846
- **Deployment:** Flask web app with a form-based prediction interface

## Project Structure

```
├── notebook/
│   ├── EDA_Customer_Churn.ipynb      # Data cleaning, EDA, feature prep
│   └── MODEL_Training.ipynb          # Model comparison & selection (prototyping)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Loads, cleans, splits raw data
│   │   ├── data_transformation.py    # Scaling, encoding, feature selection
│   │   └── model_trainer.py          # Trains, compares, and tunes models
│   ├── pipleline/
│   │   └── predict_pipeline.py       # Inference pipeline for the Flask app
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── artifacts/                        # Saved model, preprocessor, selector (generated)
└── app.py                            # Flask application entry point
```

## Workflow

1. **Data Cleaning** — `TotalCharges` was loaded as text because of a few blank rows, so
   I converted it to numeric and filled those blanks with 0 (they were all brand new
   customers with `tenure == 0`, so 0 is actually correct, not a guess). Target encoded
   to 0/1.
2. **EDA** — dug into what actually drives churn here — contract type, internet service,
   tenure, monthly charges. Month-to-month contracts and fiber optic customers churn a
   lot more, which lines up with how telecom churn usually works.
3. **Preprocessing** — `StandardScaler` on the numeric features, `OneHotEncoder` on the
   categorical ones, both combined into a single `ColumnTransformer` so it's consistent
   across train/test and the Flask app.
4. **Feature Selection** — used `SelectKBest` (ANOVA F-value) to cut the one-hot-expanded
   feature space down to the 25 most useful features.
5. **Model Comparison** — trained and compared 5 models (Logistic Regression, Decision
   Tree, Random Forest, Gradient Boosting, XGBoost). Went with F1 instead of accuracy
   since the classes are imbalanced (~73% No / ~27% Yes) — accuracy alone would've been
   misleading here.
6. **Hyperparameter Tuning** — tuned the best one (Gradient Boosting) with
   `RandomizedSearchCV`.
7. **Deployment** — saved the preprocessor, feature selector, and final model, then wired
   them into a Flask app where you fill in a customer's details and get a churn
   prediction back.

## Results

Gradient Boosting came out on top before tuning, and tuning it pushed it up a bit more:

| Model | F1 Score |
|---|---|
| Decision Tree | 0.4764 |
| XGBoost | 0.5607 |
| AdaBoost | 0.5626 |
| Random Forest | 0.5285 |
| Gradient Boosting | 0.5757 |
| **Gradient Boosting (Tuned)** | **0.5846** |

Not a huge score, but again — the point of this project was getting a real pipeline
working end to end, not maxing out the number.

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training pipeline (generates artifacts/preprocessor.pkl, selector.pkl, model.pkl)
python -m src.components.data_ingestion

# 3. Start the Flask app
python app.py
```

Then open `http://localhost:5000` in your browser, go to the prediction page, and fill in
the customer details to get a churn risk prediction.

## Tech Stack

- **Python, pandas, scikit-learn, XGBoost** — data processing and modeling
- **Flask** — web application
- **Bootstrap** — form UI

## What I'd Do Next

- Add SHAP explanations so predictions come with a "why" instead of just a label
- Try class-weighting or SMOTE to deal with the imbalance more directly
- Add proper input validation to the Flask form
- Deploy it somewhere (Render/Railway/AWS) so there's a live link, not just localhost
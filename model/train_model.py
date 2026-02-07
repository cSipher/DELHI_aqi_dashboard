# =========================
# TRAIN & SAVE XGBOOST MODEL
# =========================

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("../data/delhi_ncr_aqi_dataset.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

# -------------------------
# FEATURE ENGINEERING  ⭐ IMPORTANT
# -------------------------

# Lag Features
df['aqi_lag1'] = df['aqi'].shift(1)
df['aqi_lag3'] = df['aqi'].shift(3)

# Rolling Features
df['aqi_roll3'] = df['aqi'].rolling(3).mean()
df['aqi_roll7'] = df['aqi'].rolling(7).mean()

# Delta Features
df['aqi_delta1'] = df['aqi'] - df['aqi_lag1']
df['aqi_delta3'] = df['aqi'] - df['aqi_lag3']

# -------------------------
# TARGET VARIABLE
# -------------------------
df['severe_today'] = (df['aqi_category'] == "Severe").astype(int)
df['severe_tomorrow'] = df['severe_today'].shift(-1)

df = df.dropna()

# -------------------------
# FEATURES
# -------------------------
features = [
    'pm25','pm10','no2','so2','co','o3',
    'temperature','humidity','wind_speed','visibility',
    'aqi_lag1','aqi_lag3','aqi_roll3','aqi_roll7',
    'aqi_delta1','aqi_delta3'
]

X = df[features]
y = df['severe_tomorrow']

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, prob))

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(model, "xgb_aqi_model.pkl")

print("✅ Model Saved Successfully")

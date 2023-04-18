import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


df = pd.read_csv("https://raw.githubusercontent.com/ASTRONOE/Common_Data_Analysis/main/Kaggle/diamonds_1.csv")
df = df.drop(columns=['x', 'y', 'z', 'table', 'depth', 'volume'])
encoder = OrdinalEncoder()
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)


def rf_train():
  cat_cols = ['cut', 'color', 'clarity']
  df[cat_cols] = encoder.fit_transform(df[cat_cols])
  
  X = df.drop(columns=['log_price', 'price'])
  y = df['log_price']
  rf.fit(X, y)
  cross_validate(estimator=rf, X=X, y=y, cv=5)

@st.cache_resource   
def rf_predict_price(d):
  features = [d['cut'], d['color'], d['clarity']]
  d_encoded = encoder.transform([features])
  d_encoded_ = np.concatenate((d_encoded, [[d['carat']]]), axis=1)
  return np.exp(rf.predict(d_encoded_))[0]

def save_model(filename):
  dump(rf, filename)

def load_model(filename):
  rf = load(filename)

rf_train()
save_model('diamond_price_rf_reg.joblib')
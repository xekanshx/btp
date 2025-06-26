# ------------------ PREPROCESSING WITH RF FEATURE SELECTION ------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load preprocessed dataset with full features
df = pd.read_csv("/Users/mac/btp/preprocessed_with_power.csv")

# Drop date column if present
df = df.drop(columns=['Date'], errors='ignore')

# Rolling mean (3-day window) and lag feature for wind speed
df['WS50M_RollMean3'] = df['WS50M'].rolling(window=3).mean().fillna(method='bfill')
df['WS50M_Lag1'] = df['WS50M'].shift(1).fillna(method='bfill')

# Prepare X and y for Random Forest
X_full = df.drop(columns=['Power'])
y = df['Power']

# Fit Random Forest and get top 5 features
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_full, y)

# Get top 5 features by importance
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
top_features = feature_importance_df['feature'].iloc[:5].tolist()

# Final dataset with top 5 features + Power
df_selected = df[top_features + ['Power']]

# Scale features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

# Save to CSV
output_path = "/Users/mac/btp/reduced_scaled_top5_rf.csv"
df_scaled.to_csv(output_path, index=False)
print(f"✅ Top 5 important features selected and saved to: {output_path}")

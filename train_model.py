import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your sessions.csv file
df = pd.read_csv('data/sessions.csv')

# Preprocessing
df['date_start'] = pd.to_datetime(df['date_start'])
df['date_end'] = pd.to_datetime(df['date_end'])
df['duration_mins'] = (df['date_end'] - df['date_start']).dt.total_seconds() / 60

# Feature Engineering
df['start_hour'] = df['date_start'].dt.hour
df['day_of_week'] = df['date_start'].dt.dayofweek

# One-hot encoding for session types
df_encoded = pd.get_dummies(df, columns=['session_type'])

# Define the exact features we want to keep
features = ['start_hour', 'day_of_week', 'session_type_Practice', 'session_type_Qualifying', 'session_type_Race']
X = df_encoded[features]
y = df_encoded['duration_mins']

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'data/f1_predictor.pkl')
print("Model trained successfully with feature names!")
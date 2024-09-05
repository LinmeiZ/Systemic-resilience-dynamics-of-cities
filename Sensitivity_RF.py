# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_excel('Data\\Sensitivity_Analysis_Dataset_Test.xlsx')

# Split the data into features and target
X_RF = data.drop('Y', axis=1)
y_RF = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X_RF, y_RF, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train_scaled, y_train)  # Ensure you are fitting the scaled data

y_pred = model.predict(X_test_scaled)  # Ensure predictions are on scaled data


# Sensitivity Analysis
original_predictions = model.predict(X_test_scaled)
feature_stds = np.std(X_train_scaled, axis=0)
sensitivity_scores = []

for i in range(X_train.shape[1]):
    X_test_disturbed = X_test_scaled.copy()
    X_test_disturbed[:, i] += feature_stds[i]  # Use numpy array indexing
    disturbed_predictions = model.predict(X_test_disturbed)
    prediction_differences = disturbed_predictions - original_predictions
    S_M = np.mean(prediction_differences) / np.std(y_train) * 100
    sensitivity_scores.append(S_M)

# Optional: Plot the sensitivity scores
features = X_RF.columns
plt.barh(features, sensitivity_scores)
plt.xlabel('Sensitivity Score (%)')
plt.title('Feature Sensitivity Analysis')
plt.show()


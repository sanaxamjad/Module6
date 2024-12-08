# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Explore Dataset
data = pd.read_csv('bitcoinalpha.csv')
print("Dataset Overview:")
print(data.head())
print("Columns in the dataset:", data.columns)

# Rename columns to match required format
data.rename(columns={
    'SOURCE': 'from',
    'TARGET': 'to',
    'TIME': 'time',
    'RATING': 'rating'
}, inplace=True)

# Verify necessary columns exist
required_columns = ['from', 'to', 'time', 'rating']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(data.info())

# Feature Engineering
data['from_activity'] = data.groupby('from')['from'].transform('count')
data['to_activity'] = data.groupby('to')['to'].transform('count')
data['transaction_frequency'] = data.groupby('from')['time'].transform(lambda x: x.diff().fillna(0))
data['average_rating_given'] = data.groupby('from')['rating'].transform('mean')
data['average_rating_received'] = data.groupby('to')['rating'].transform('mean')

# Drop unnecessary columns
features = ['from_activity', 'to_activity', 'transaction_frequency', 'average_rating_given', 'average_rating_received']
X = data[features].fillna(0)
y = data['rating']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Identify Misclassified Samples
errors = np.abs(predictions - y_test)
error_df = pd.DataFrame(X_test, columns=features)
error_df['actual'] = y_test.values
error_df['predicted'] = predictions
error_df['error'] = errors

# Top 5 Errors
largest_errors = error_df.sort_values(by='error', ascending=False).head(5)
print("Top 5 Misclassified Samples:")
print(largest_errors)

# Visualizations
# 1. Distribution of Prediction Errors
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=20, kde=True, color='blue')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# 2. Actual vs Predicted Ratings (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

# 3. Feature Importance (Bar Plot)
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# 4. Residuals vs Predicted (Residual Plot)
residuals = y_test - predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions, y=residuals, color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Ratings')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()

# 5. Correlation Heatmap of Features
correlation_matrix = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# 6. Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.histplot(data['rating'], bins=20, kde=True, color='orange')
plt.title('Distribution of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.show()

# 7. Prediction Error by Rating
error_by_rating = pd.DataFrame({'actual': y_test, 'predicted': predictions, 'error': errors})
plt.figure(figsize=(10, 6))
sns.boxplot(x='actual', y='error', data=error_by_rating, palette='Set2')
plt.title('Prediction Error by Rating Value')
plt.xlabel('Actual Rating')
plt.ylabel('Error (Predicted - Actual)')
plt.show()

# Save Results
largest_errors.to_csv('largest_errors.csv', index=False)

# Limitations
print("\nLimitations of the Analysis:")
print("1. Dataset is imbalanced with skewed positive ratings.")
print("2. Features are limited to transactional data without external context.")
print("3. Potential biases in the dataset might influence model predictions.")

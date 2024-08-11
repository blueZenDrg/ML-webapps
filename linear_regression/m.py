import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('housing.csv')

# Separate features and target
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']




# Handle categorical variable
X = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Split the data
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle categorical variable
if 'ocean_proximity' in X_train.columns:
    X_train = pd.get_dummies(X_train, columns=['ocean_proximity'], drop_first=True)
if 'ocean_proximity' in X_test.columns:
    X_test = pd.get_dummies(X_test, columns=['ocean_proximity'], drop_first=True)


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {'RMSE': rmse, 'R2': r2}

# Print results
print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")

# Feature importance for Linear Regression
lr_model = models['Linear Regression']
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': lr_model.coef_})
feature_importance = feature_importance.sort_values('importance', key=abs, ascending=False)

print("\nTop 5 Important Features (Linear Regression):")
print(feature_importance.head())

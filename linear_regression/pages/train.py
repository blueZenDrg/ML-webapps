import streamlit as st
from sklearn.linear_model import LinearRegression, BayesianRidge, QuantileRegressor, OrthogonalMatchingPursuit, LassoLars
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

def train(data):
    st.subheader('Train Regression Models')

    features = ['RAM', 'ROM', 'Mobile_Size', 'Battery_Power']
    X = data[features]
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Ordinary Least Squares': (LinearRegression(), {}),
        'Simple Linear Regression': (LinearRegression(), {}),
        'Multiple Linear Regression': (LinearRegression(), {}),
        'Polynomial Regression': (make_pipeline(PolynomialFeatures(), LinearRegression()), 
                                  {'polynomialfeatures__degree': [2, 3, 4]}),
        'Orthogonal Matching Pursuit': (OrthogonalMatchingPursuit(), 
                                        {'n_nonzero_coefs': [1, 5, 10, 20]}),
        'Bayesian Regression': (BayesianRidge(), 
                                {'alpha_1': [1e-6, 1e-5, 1e-4], 'alpha_2': [1e-6, 1e-5, 1e-4]}),
        'Quantile Regression': (QuantileRegressor(), 
                                {'quantile': [0.25, 0.5, 0.75], 'alpha': [0.1, 1.0, 10.0]}),
        'Isotonic Regression': (IsotonicRegression(out_of_bounds='clip'), {}),
        'Least-angle Regression (LARS)': (LassoLars(), 
                                          {'alpha': [0.01, 0.1, 1.0, 10.0]}),
        'Random Forest Regressor': (RandomForestRegressor(),
                                    {'n_estimators': [100, 200, 300],
                                     'max_depth': [None, 10, 20, 30],
                                     'min_samples_split': [2, 5, 10]})
    }

    selected_model = st.selectbox('Select Regression Model', list(models.keys()))

    if st.button('Train Model'):
        model, param_grid = models[selected_model]

        if selected_model == 'Simple Linear Regression':
            X_train_simple = X_train[['RAM']].values
            X_test_simple = X_test[['RAM']].values
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_simple, y_train)
            y_pred = grid_search.predict(X_test_simple)
        elif selected_model == 'Isotonic Regression':
            X_train_isotonic = X_train['RAM'].values.reshape(-1, 1)
            X_test_isotonic = X_test['RAM'].values.reshape(-1, 1)
            model.fit(X_train_isotonic, y_train)
            y_pred = model.predict(X_test_isotonic)
        else:
            grid_search = GridSearchCV(make_pipeline(StandardScaler(), model), param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = 1 - (mae / y_test.mean())

        st.write(f'Model: {selected_model}')
        if selected_model != 'Isotonic Regression':
            st.write(f'Best Parameters: {grid_search.best_params_}')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'R-squared Score: {r2:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.write(f'Accuracy: {accuracy:.2f}')

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)

import streamlit as st
from sklearn.model_selection import cross_val_score
import pandas as pd

def cross_validation_page():
    from sklearn.datasets import load_iris

    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    st.header("Cross-Validation")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.write("No model has been trained yet. Please train a model first.")
    else:
        # Select features
        st.write("### Select Features for Cross-Validation")
        selected_features = st.multiselect("Select features to include in the model", iris.feature_names, default=iris.feature_names)
        X = iris_df[selected_features]

        # Select target variable
        st.write("### Select Target Variable")
        target_variable = st.selectbox("Select the target variable", iris.feature_names + ['species'])
        # Convert the continuous target variable to a categorical variable
        if target_variable == 'species':
            y = iris_df['species']
        else:
            y = pd.cut(iris_df[target_variable], bins=3, labels=[0, 1, 2])

        # Cross-validation
        st.write("### Cross-Validation Results")
        model_type = st.session_state.model.__class__.__name__
        if model_type == "LogisticRegression":
            scores = cross_val_score(st.session_state.model, X, y, cv=5)
        elif model_type == "SVC":
            scores = cross_val_score(st.session_state.model, X, y, cv=5)
        elif model_type == "RandomForestClassifier":
            scores = cross_val_score(st.session_state.model, X, y, cv=5)
        elif model_type == "GradientBoostingClassifier":
            scores = cross_val_score(st.session_state.model, X, y, cv=5)

        st.write(f"<b>Cross-validation accuracy scores:</b> {scores}", unsafe_allow_html=True)
        st.write(f"<b>Average cross-validation accuracy:</b> {scores.mean():.2f}", unsafe_allow_html=True)

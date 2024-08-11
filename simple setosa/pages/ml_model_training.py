import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def ml_model_training_page():
    from sklearn.datasets import load_iris

    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    st.header("ML Model Training")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)

    # Feature selection
    st.write("### Select Target Variable")
    target_variable = st.selectbox("Select the target variable", iris.feature_names + ['species'])

    # Select features
    st.write("### Select Features for Training")
    selected_features = st.multiselect("Select features to include in the model", iris.feature_names, default=iris.feature_names)

    # Model selection
    st.write("### Select a Model")
    model_type = st.selectbox("Choose a model", ["Logistic Regression", "SVM", "Random Forest", "Gradient Boosting"])

    # Train the model
    if st.button("Train Model"):
        st.markdown("<hr style='border:1px dashed #2c3e50;'>", unsafe_allow_html=True)
        st.info("Training the model... Please wait.")
        
        X = iris_df[selected_features]
        if target_variable == 'species':
            y = iris_df['species']
        else:
            # Convert the continuous target variable to a categorical variable
            y = pd.cut(iris_df[target_variable], bins=3, labels=[0, 1, 2])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model
        if model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_type == "SVM":
            model = SVC(random_state=42)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Save the model in session state
        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.report = classification_report(y_test, predictions, output_dict=True)

        st.success(f"Model trained successfully with an accuracy of {accuracy:.2f}.")
        st.balloons()

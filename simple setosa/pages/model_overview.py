import streamlit as st
import pandas as pd
import joblib

def model_overview_page():
    st.header("Model Overview")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.write("No model has been trained yet. Please train a model first.")
    else:
        st.write(f"### Model Accuracy: {st.session_state.accuracy:.2f}")
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(st.session_state.report).transpose().style.set_properties(
            **{'background-color': '#e0e0e0', 'color': '#2c3e50'}
        ))

        # Option to download the trained model
        if st.button("Download Model"):
            joblib.dump(st.session_state.model, 'iris_model.pkl')
            st.success("Model saved as `iris_model.pkl`.")

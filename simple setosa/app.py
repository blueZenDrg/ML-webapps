import streamlit as st
from pages.dataset_overview import dataset_overview_page
from pages.eda import eda_page
from pages.ml_model_training import ml_model_training_page
from pages.model_overview import model_overview_page
from pages.cross_validation import cross_validation_page
from pages.about import about_page

# Global CSS styles
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 10px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .stApp {
            max-width: 80%;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app title with a background image
st.markdown("""
    <div style='text-align:center; padding: 10px; background: #2980b9; color: white; border-radius: 10px;'>
        <h1 style='margin-bottom:5px;'>Iris Dataset Explorer</h1>
        <p style='margin-top:5px;'>A Beautiful Visualization & Machine Learning Tool</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                            ["Dataset Overview", "EDA", "ML Model Training", 
                             "Model Overview", "Cross-Validation", "About"])

# Page navigation
if page == "Dataset Overview":
    dataset_overview_page()
elif page == "EDA":
    eda_page()
elif page == "ML Model Training":
    ml_model_training_page()
elif page == "Model Overview":
    model_overview_page()
elif page == "Cross-Validation":
    cross_validation_page()
elif page == "About":
    about_page()

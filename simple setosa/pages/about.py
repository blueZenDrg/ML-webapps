import streamlit as st

def about_page():
    st.header("About This App")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x200.png?text=Iris+Dataset+Explorer", use_column_width=True)

    st.write("""
    This Streamlit app provides an interactive exploration of the Iris dataset.
    It allows users to view the dataset, perform simple exploratory data analysis (EDA),
    train a machine learning model, and visualize the model's performance.
    The app is designed for educational purposes to demonstrate basic data science and 
    machine learning techniques using a well-known dataset.
    """)

    st.markdown("<hr style='border:1px solid #2980b9;'>", unsafe_allow_html=True)

    st.write("""
    **Features:**
    - **Dataset Overview:** Get a quick glimpse of the dataset and summary statistics.
    - **Exploratory Data Analysis (EDA):** Visualize the relationships between features and explore feature distributions.
    - **ML Model Training:** Train a machine learning model using various algorithms.
    - **Model Overview:** Review the model's performance and download the trained model.
    - **Cross-Validation:** Perform cross-validation to assess model generalization.
    """)

    st.markdown("<br><i>Created with 💻 by Mohanarangan</i>", unsafe_allow_html=True)

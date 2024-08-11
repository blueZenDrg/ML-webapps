import streamlit as st
import pandas as pd

def dataset_overview_page():
    from sklearn.datasets import load_iris
    
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    st.header("Iris Dataset Overview")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)

    # Display dataset with adjustable number of rows
    st.write("### Dataset")
    num_rows = st.slider("Number of rows to view", min_value=1, max_value=len(iris_df), value=10)
    st.dataframe(iris_df.head(num_rows).style.set_properties(**{'background-color': '#f4f4f4', 'color': '#2c3e50'}))

    # Show summary statistics
    if st.checkbox("Show summary statistics"):
        st.write("### Summary Statistics")
        st.write(iris_df.describe().style.set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#2980b9'), ('color', 'white')]}]
        ))

    # Show dataset distribution by species
    st.write("### Distribution by Species")
    st.bar_chart(iris_df['species'].value_counts(), use_container_width=True)

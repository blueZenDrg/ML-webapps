import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def eda_page():
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("<hr style='border:2px solid #2c3e50;'>", unsafe_allow_html=True)

    # Pairplot for feature relationships
    st.write("### Pairplot of Features")
    st.markdown("This pairplot shows the relationships between different features, categorized by species.")
    sns.pairplot(iris_df, hue="species", markers=["o", "s", "D"])
    st.pyplot(plt.gcf())

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature distribution by species
    st.write("### Feature Distribution by Species")
    feature = st.selectbox("Select a feature", iris.feature_names)
    fig, ax = plt.subplots()
    sns.histplot(data=iris_df, x=feature, hue="species", kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_df = iris_df.drop(columns=['species'])  # Exclude the 'species' column
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

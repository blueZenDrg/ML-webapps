import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def corr_heat(data):
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def pairplot(data):
    st.subheader('Pairplot')
    fig = sns.pairplot(data, vars=['RAM', 'ROM', 'Mobile_Size', 'Battery_Power', 'Price'])
    st.pyplot(fig)


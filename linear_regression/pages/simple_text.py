import streamlit as st

def Home():
    st.title('Mobile Phone Price Analysis')
    st.write('Welcome to the Mobile Phone Price Analysis app. Use the sidebar to navigate through different pages.')


def describe(data):
    st.subheader('Basic Statistics')
    st.write(data.describe())

def RawData(data):
    st.title('Raw Data')
    st.write(data.head())
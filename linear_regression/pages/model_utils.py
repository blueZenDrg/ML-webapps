import streamlit as st
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(data):
    st.subheader('Feature Importance')
    features = ['RAM', 'ROM', 'Mobile_Size', 'Battery_Power']
    X = data[features]
    y = data['Price']
    model = LinearRegression()
    model.fit(X, y)
    importance = pd.DataFrame({'feature': features, 'importance': model.coef_})
    importance = importance.sort_values('importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='feature', data=importance, ax=ax)
    st.pyplot(fig)

def price_prediction(data):
    st.subheader('Predict Phone Price')
    ram = st.slider('RAM (GB)', 1, 12, 4)
    rom = st.slider('ROM (GB)', 4, 256, 64)
    mobile_size = st.slider('Mobile Size (inches)', 4.0, 7.0, 5.5)
    battery_power = st.slider('Battery Power (mAh)', 1000, 6000, 3000)

    features = ['RAM', 'ROM', 'Mobile_Size', 'Battery_Power']
    X = data[features]
    y = data['Price']
    model = LinearRegression()
    model.fit(X, y)

    if st.button('Predict Price'):
        prediction = model.predict([[ram, rom, mobile_size, battery_power]])
        st.write(f'Predicted Price: ${prediction[0]:,.2f}')

def price_distribution(data):
    st.subheader('RAM vs Price')
    fig, ax = plt.subplots()
    sns.scatterplot(x='RAM', y='Price', data=data, ax=ax)
    st.pyplot(fig)

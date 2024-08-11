import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, QuantileRegressor, OrthogonalMatchingPursuit, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from pages.simple_text import *
from pages.vis import *
from pages.train import train
from pages.model_utils import *

def vs_plot(data, x, y):
    st.subheader(f'{x} vs {y}')
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, data=data, ax=ax)
    st.pyplot(fig)

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

data = load_data()

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Raw Data', 'Basic Statistics', 'Correlation Heatmap', 'Pairplot', 'Linear Regression', 'Feature Importance', 'Price Prediction', 'Price Distribution', 'RAM vs Price', 'Mobile Size vs Price', 'Train'])

if page == 'Home':
    Home()
elif page == 'Raw Data':
    RawData(data)
elif page == 'Basic Statistics':
    describe(data)
elif page == 'Correlation Heatmap':
    corr_heat(data)
elif page == 'Pairplot':
    pairplot(data)
elif page == 'Train':
    train(data)
elif page == 'Feature Importance':
    plot_feature_importance(data)
elif page == 'Price Prediction':
    price_prediction(data)
elif page == 'Price Distribution':
    price_distribution(data)

elif page == 'RAM vs Price':
    vs_plot(data, 'RAM', 'Price')

elif page == 'Mobile Size vs Price':
    vs_plot(data, 'Mobile_Size', 'Price')


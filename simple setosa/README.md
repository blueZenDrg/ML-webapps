# Iris Dataset Explorer

## Overview

The Iris Dataset Explorer is a Streamlit web application that provides an interactive exploration of the famous Iris dataset. This app allows users to view the dataset, perform exploratory data analysis (EDA), train machine learning models, and evaluate model performance.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/)

## Features

- **Dataset Overview**: View the Iris dataset and summary statistics
- **Exploratory Data Analysis (EDA)**: Visualize feature relationships and distributions
- **ML Model Training**: Train various machine learning models on the Iris dataset
- **Model Overview**: Review model performance and download trained models
- **Cross-Validation**: Perform cross-validation to assess model generalization
- **About**: Information about the application and its purpose

## File Structure

- `app.py`: Main application file with Streamlit setup and navigation
- `utils/`:
  - `data_utils.py`: Functions for loading and preprocessing the Iris dataset
  - `model_utils.py`: Utility functions for model training and evaluation
- `pages/`:
  - `about.py`: About page content
  - `cross_validation.py`: Cross-validation functionality
  - `dataset_overview.py`: Dataset overview page
  - `eda.py`: Exploratory Data Analysis page
  - `ml_model_training.py`: Machine Learning model training page
  - `model_overview.py`: Model performance overview page


## How to Run

1. Install the required dependencies (Streamlit, Pandas, Scikit-learn, Seaborn, Matplotlib)
2. Run the application using the command: `streamlit run app.py`

## Technologies Used

- Streamlit
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Author

Created by Mohanarangan

Enjoy exploring the Iris Dataset with this interactive tool!

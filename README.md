# House Recommendation and Price Estimation System

This is our Final Project for CPSC 458: Automated Decision Systems

## Members
Daniel Metaferia ()
Vimbisai Basvi ()
Aileen Siele (acs274)
## Overview
Our project is an advanced analytical tool designed to streamline the home-buying process by leveraging cutting-edge machine learning algorithms. This system offers precise house price estimations and personalized recommendations. Initially we experimented with various models like Random Forest and Decision Trees, and we ended up focusing on using Linear Regression and clustering (KMeans) for the final implementation due to their enhanced predictive performance.

### Features
- **Price Estimation with Linear Regression:** Employs Linear Regression to estimate house prices effectively, emphasizing the impact of bedrooms and bathrooms on pricing.
  
- **Advanced Clustering for Customized Recommendations:** Utilizes KMeans clustering to group houses based on shared features, aligning recommendations with user preferences and budget.

## Data Processing
Our project uses a synthetic dataset of houses based in the UK and initially, it comprised only the longitude and latitude in addition to the other columns. We had to use the Geocoding API to convert each of the 30,000 houses' latitudes and longitudes to their respective locations. Using this information, we generated a 'cities' column. This data enhancement was pivotal for our subsequent analyses.

## Challenges Faced
- **Data Synthesis and Enrichment:** Transforming a basic dataset into a comprehensive one required significant effort, particularly in generating usable city information from geographic coordinates.
- **Model Exploration and Optimization:** We experimented with four different machine learning models. However, due to unsatisfactory prediction rates, we opted to proceed with only two—Linear Regression and Clustering. Despite attempts to integrate categorical data via one-hot encoding, this method proved ineffectual, leading us to focus solely on numerical data especially for linear regression.
- **Learning Curve:** Our journey into AI was fueled by self-learning, progressing from one resource to another. This exploratory process is detailed in our Jupyter notebooks, which details our progression from having limited knowledge of machine learning to developing a functional predictive system.

## Model Performance
- **Linear Regression:** We achieved the most reliable price estimations, with an R² value indicating that a significant percentage of price variability is explained by the selected features (bedrooms and bathrooms). 
- **Decision Trees and Random Forests:** These models were initially considered but ultimately not included in the final system due to less accurate predictions.

## How to Run the Program
- **Prerequisites:** Install Python (version 3.8 or newer recommended) and required libraries using `pip install -r requirements.txt`.
- **Repository:** Clone the repository with `git clone https://github.com/aileensiele/HouseRecommendation.git`.
- **Execution:** Navigate to the directory and start the application with `python main.py`.
- **Interaction:** The website has user-friendly input buttons for users to select their desired features in a house. The user can then select whether they would like a price estimation or house recommendation based on the features provided. The recommendations also have explanations.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_and_predict_price(bedrooms, bathrooms, city, df):
    # Load the datasets something like:
    # df = pd.read_csv("../updated_with_cities.csv")

    # Filter the dataset based on the location
    filtered_df = df[df['city'] == city]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        return "No data available for the specified location."

    # Select predictors and the response variable from the filtered data
    X = filtered_df[['bedrooms', 'bathrooms']]
    y = filtered_df['price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for the input features to make a prediction
    input_data = pd.DataFrame(
        {'bedrooms': [bedrooms], 'bathrooms': [bathrooms]})

    # Predict the price
    predicted_price = model.predict(input_data)[0]
    # Get model coefficients and intercept
    intercept = model.intercept_
    coeff_bedrooms, coeff_bathrooms = model.coef_

    # Explain the contribution of each component
    base_price = intercept
    added_from_bedrooms = coeff_bedrooms * bedrooms
    added_from_bathrooms = coeff_bathrooms * bathrooms

    explanation = f"""
    Predicted Price Breakdown:
    Base Price (Intercept): ${base_price:,.2f}
    Added from {bedrooms} Bedrooms: ${added_from_bedrooms:,.2f} (${coeff_bedrooms:,.2f} each)
    Added from {bathrooms} Bathrooms: ${added_from_bathrooms:,.2f} (${coeff_bathrooms:,.2f} each)
    Total Predicted Price: ${predicted_price:,.2f}
    """
    print(explanation)
    return predicted_price, explanation

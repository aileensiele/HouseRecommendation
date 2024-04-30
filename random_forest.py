import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # One-hot encode categorical variables
    categorical_columns = ['house_type', 'neighbourhood_quality', 'house_condition', 'crime_rate']
    encoder = OneHotEncoder()
    transformer = ColumnTransformer([('one_hot', encoder, categorical_columns)], remainder='passthrough')
    df_encoded = transformer.fit_transform(df)
    
    # Accessing categories from the fitted transformer
    fitted_encoder = transformer.named_transformers_['one_hot']
    categories = fitted_encoder.categories_

    # Creating new column names for the categorical variables
    category_mapping = [f"{cat}__{subcat}" for cat, sublist in zip(categorical_columns, categories) for subcat in sublist]

    # Get names of columns that were not transformed (passthrough)
    passthrough_indices = [i for i, col in enumerate(df.columns) if col not in categorical_columns]
    passthrough_columns = [df.columns[i] for i in passthrough_indices]

    # Combine all column names
    all_columns = category_mapping + passthrough_columns

    # Create the DataFrame with the appropriate column names
    processed_df = pd.DataFrame(df_encoded, columns=all_columns)

    # Log-transform the 'price' column
    processed_df['log_price'] = np.log(df['price'])

    # Log-transform the 'garden_size' column if necessary
    processed_df['log_garden_size'] = np.log(df['garden_size'])

    # Drop unnecessary columns
    processed_df.drop(columns=['Unnamed: 0', 'price', 'house_keywords', 'garden_size'], inplace=True)

    return processed_df

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_price(user_input, model, column_order):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[column_order]
    predicted_log_price = model.predict(input_df)
    predicted_price = np.exp(predicted_log_price[0])
    return predicted_price

def explanation(model):
    return None

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('synthetic_house_recommendation_data.csv')

    # Preprocess data
    processed_df = preprocess_data(df)

    # Define features and target variable
    X = processed_df.drop(columns=['log_price'], axis=1)
    y = processed_df['log_price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions on test set
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Define user input
    user_input = {
        # Define user input here
    }

    # Predict price based on user input
    predicted_price = predict_price(user_input, model, X_train.columns)
    print(f'Predicted Price: ${predicted_price:.2f}')

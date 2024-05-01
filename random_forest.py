import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Apply transformation and convert to DataFrame immediately
    df_encoded = transformer.fit_transform(df)
    df_encoded = pd.DataFrame(df_encoded, columns=transformer.get_feature_names_out())
    
    # Convert all to int/float for binary and numerical columns
    binary_columns = [col for col in df_encoded.columns if "one_hot" in col]  # Adjusted to filter by transformer name
    df_encoded[binary_columns] = df_encoded[binary_columns].astype(int)

    # Correct the naming for the 'remainder' columns and convert data types
    for col in df.columns:
        if col not in categorical_columns:
            new_col_name = col.replace("remainder__", "")  # Remove the prefix added by ColumnTransformer
            df_encoded.rename(columns={f"remainder__{col}": new_col_name}, inplace=True)
            if df_encoded[new_col_name].dtype == 'object':
                df_encoded[new_col_name] = pd.to_numeric(df_encoded[new_col_name], errors='coerce')

    df_encoded.fillna(df_encoded.mean(), inplace=True)

    # Log-transform the 'price' column
    df_encoded['log_price'] = np.log(df_encoded['price'])

    # # Log-transform the 'garden_size' column if necessary
    df_encoded['log_garden_size'] = np.log(df_encoded['garden_size'] + 1)  # +1 to handle zero garden size

    # Drop unnecessary columns
    df_encoded.drop(columns=['Unnamed: 0', 'price', 'garden_size','house_keywords'], inplace=True) #cld add garden_size in exchange for log_garden_size; price for log_price

    return df_encoded

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_price(user_input, model, column_order):
    input_df = pd.DataFrame([user_input], columns=column_order)  # Ensure column order matches
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
    print(processed_df.info())

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions on test set
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Define user input
    user_input = {
        'one_hot__house_type_Bungalow': 0,
        'one_hot__house_type_Detached': 1,
        'one_hot__house_type_Flat': 0,
        'one_hot__house_type_Land': 0,
        'one_hot__house_type_Park Home': 0,
        'one_hot__house_type_Semi-Detached': 0,
        'one_hot__house_type_Terraced': 0,
        'one_hot__neighbourhood_quality_Abysmal': 0,
        'one_hot__neighbourhood_quality_Alright': 0,
        'one_hot__neighbourhood_quality_Outstanding': 1,
        'one_hot__neighbourhood_quality_Pleasant': 0,
        'one_hot__neighbourhood_quality_Rough': 0,
        'one_hot__house_condition_Abysmal': 0,
        'one_hot__house_condition_Alright': 0,
        'one_hot__house_condition_Outstanding': 1,
        'one_hot__house_condition_Pleasant': 0,
        'one_hot__house_condition_Rough': 0,
        'one_hot__crime_rate_Extreme': 0,
        'one_hot__crime_rate_High': 0,
        'one_hot__crime_rate_Low': 1,
        'one_hot__crime_rate_Medium': 0,
        'one_hot__crime_rate_Non Existent': 0,
        'one_hot__crime_rate_Very High': 0,
        'one_hot__crime_rate_Very Low': 0,
        'latitude': 40.7128,
        'longitude': -74.0060,
        
        'bedrooms': 3,
        'bathrooms': 2,
        'log_garden_size': np.log(500.0)  # Example garden size in square feet
    }
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    input_df = input_df[X_train.columns]  # Ensure column order matches the training data

    # If you are running this in your full script, assuming 'model' and 'X_train' are defined as in your script
    predicted_log_price = model.predict(input_df)
    predicted_price = np.exp(predicted_log_price[0])
    

    print(f'Predicted Price: ${predicted_price:.2f}')
    
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(feature_importances.nlargest(10).plot(kind='barh'))
    plt.title('Top 10 Important Features')

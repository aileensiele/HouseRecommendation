import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


def load_and_preprocess_data():
    # Load the dataset
    data_path = 'updated_with_cities.csv'
    df = pd.read_csv(data_path)
    df.drop(['Unnamed: 0', 'latitude', 'longitude', 'house_keywords'], axis=1, inplace=True)

    # Replace '\N' with NaN
    df.replace('\\N', np.nan, inplace=True)

    # Handle missing values if any
    df.fillna({
        'bedrooms': df['bedrooms'].median(),
        'bathrooms': df['bathrooms'].median(),
        'garden_size': df['garden_size'].median(),
        'house_type': df['house_type'].mode()[0],
        'neighbourhood_quality': df['neighbourhood_quality'].mode()[0],
        'house_condition': df['house_condition'].mode()[0],
        'crime_rate': df['crime_rate'].mode()[0],
        'city': df['city'].mode()[0]
    }, inplace=True)

    # Convert categorical columns to type 'category' if not already
    categorical_cols = ['house_type', 'neighbourhood_quality', 'house_condition', 'crime_rate', 'city']
    for col in categorical_cols:
        if df[col].dtype != 'category':
            df[col] = df[col].astype('category')

    return df, categorical_cols


def predict_price(df, categorical_cols, user_preferences):
    # values for each feature
    default_values = {
        'bedrooms': df['bedrooms'].median(),
        'bathrooms': df['bathrooms'].median(),
        'garden_size': df['garden_size'].median(),
        'house_type': df['house_type'].mode()[0],
        'neighbourhood_quality': df['neighbourhood_quality'].mode()[0],
        'house_condition': df['house_condition'].mode()[0],
        'crime_rate': df['crime_rate'].mode()[0],
        'city': df['city'].mode()[0]
    }

    # Merge user preferences with default values
    complete_preferences = {**default_values, **user_preferences}

    # Prepare the pipeline
    numerical_cols_regression = ['bedrooms', 'bathrooms', 'garden_size']
    column_transformer_regression = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols_regression)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', column_transformer_regression),
        ('regressor', DecisionTreeRegressor(max_depth=5,random_state=42))
    ])

    # train the model
    X = df[categorical_cols + numerical_cols_regression]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # predict using merged preferences
    user_df = pd.DataFrame([complete_preferences])
    predicted_price = pipeline.predict(user_df)
    return predicted_price[0]

    # # Perform cross-validation
    # scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    # rmse_scores = np.sqrt(-scores)

    # # Train the model to predict using specific user preferences
    # pipeline.fit(X, y)  # Train on the full dataset for actual prediction
    # user_df = pd.DataFrame([complete_preferences])
    # predicted_price = pipeline.predict(user_df)
    # return predicted_price[0], rmse_scores.mean()


def perform_clustering(df, categorical_cols):
    numerical_cols_clustering = ['bedrooms', 'bathrooms', 'garden_size', 'price']

    column_transformer_clustering = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols_clustering)
        ],
        remainder='passthrough'
    )

    pipeline_clustering = Pipeline([
        ('preprocessor', column_transformer_clustering),
        ('cluster', KMeans(n_clusters=5, random_state=42))
    ])

    X_clustering = df[categorical_cols + numerical_cols_clustering]
    df['Cluster'] = pipeline_clustering.fit_predict(X_clustering)
    return pipeline_clustering

def format_explanation(explanation):
    """
    Format the explanation dictionary into a more user-friendly text output, without '\n' and with proper new lines in code.
    """
    return (
        f"Based on your preferences for {explanation['input_preferences']}, with specific requirements on {explanation['strict_criteria']}, "
        f"we looked for homes that match your lifestyle and budget. We first grouped homes into similar clusters to find communities that might appeal to you. "
        f"Then, we focused on homes in your preferred city that fit your budget. "
        f"We prioritized homes that matched your type preference ('{explanation['house_type']}') and scored each option based on how well it met your needs. "
        f"The best matches were sorted by how closely they met your preferences and their proximity to your budget limit, helping you find the best value within your range. "
        f"The top recommendations were chosen based on these criteria, aiming to find the perfect fit for your needs within {explanation['city']}."
    )

def recommend_houses(df, user_preferences, pipeline_clustering):
    default_values = {
        'bedrooms': df['bedrooms'].median(),
        'bathrooms': df['bathrooms'].median(),
        'house_type': df['house_type'].mode()[0],
        'city': df['city'].mode()[0],
        'neighbourhood_quality': df['neighbourhood_quality'].mode()[0],
        'house_condition': df['house_condition'].mode()[0],
        'crime_rate': df['crime_rate'].mode()[0],
        'garden_size': df['garden_size'].median(),
        'price': df['price'].median()
    }

    strict_criteria = user_preferences.pop('strict', [])
    complete_preferences = {**default_values, **user_preferences}
    budget = complete_preferences.pop('budget')

    # Transform user's preferences into the format used by the model
    user_df = pd.DataFrame([complete_preferences])
    user_df_transformed = pipeline_clustering['preprocessor'].transform(user_df)
    user_cluster = pipeline_clustering['cluster'].predict(user_df_transformed)[0]

    # Build the query based on user cluster and budget
    query = (df['Cluster'] == user_cluster) & (df['price'] <= budget) & (df['city'] == complete_preferences['city'])
    for criterion in strict_criteria:
        if criterion in complete_preferences and complete_preferences[criterion] is not None:
            query &= (df[criterion] == complete_preferences[criterion])

    recommended_houses = df[query].copy()
    recommended_houses['score'] = (recommended_houses['house_type'] == complete_preferences['house_type']).astype(int)

    explanation = {
        'input_preferences': ', '.join(f"{k}: {v}" for k, v in user_preferences.items() if v is not None),
        'strict_criteria': ', '.join(strict_criteria),
        'house_type': complete_preferences['house_type'],
        'city': complete_preferences['city']
    }

    if recommended_houses.empty:
        recommended_houses = df[(df['price'] <= budget) & (df['city'] == complete_preferences['city'])].copy()
        recommended_houses['score'] = 0

    if not recommended_houses.empty:
        recommended_houses['budget_distance'] = budget - recommended_houses['price']
        recommended_houses = recommended_houses.sort_values(by=['score', 'budget_distance'], ascending=[False, True]).head(10)
        return recommended_houses[['bedrooms', 'bathrooms', 'house_type', 'city', 'price', 'score']], format_explanation(explanation)
    else:
        return "No houses found in " + complete_preferences['city'] + " within your budget.", {}

def main():
    df, categorical_cols = load_and_preprocess_data()
    pipeline_clustering = perform_clustering(df, categorical_cols)
    user_preferences = {
        'bedrooms': 3,
        'bathrooms': 2,
        'city': 'London',
        'house_type': 'Detached',
        'budget': 500000,
        'neighbourhood_quality': 'Pleasant',
        'house_condition': 'Pleasant',
        'crime_rate': 'Low',
        'strict': ['city', 'neighbourhood_quality', 'crime_rate']
    }

    # user_preferences = {
    #     'bedrooms': 4, 'bathrooms': 2, 'city': 'London', 'house_type': 'Detached', 'budget': 200000,
    #     'strict': ['city', 'bedrooms', 'bathrooms']
    # }
    price_estimate = predict_price(df, categorical_cols, user_preferences)
    formatted_price = f"£{price_estimate:,.2f}"  # Formatting to two decimal places with comma as a thousand separator
    print(f"Estimated price for specified house: {formatted_price}")
    # price_estimate, average_rmse = predict_price(df, categorical_cols, user_preferences)
    # print(f"Estimated price for specified house:f £{price_estimate:,.2f}")
    # print(f"Average RMSE from cross-validation: £{average_rmse:,.2f}")

    recommended_houses, explanation = recommend_houses(df, user_preferences, pipeline_clustering)
    print("Recommended houses based on your preferences:")
    print(recommended_houses)
    print("\n")
    print(explanation)


if __name__ == "__main__":
    main()

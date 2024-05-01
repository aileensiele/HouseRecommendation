import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

def predict_prices(df, categorical_cols):
    numerical_cols_regression = ['bedrooms', 'bathrooms', 'garden_size'] 

    # ColumnTransformer for Regression
    column_transformer_regression = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols_regression)
        ],
        remainder='passthrough'
    )

    # Regression Pipeline
    pipeline_regression = Pipeline([
        ('preprocessor', column_transformer_regression),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

    X_regression = df[categorical_cols + numerical_cols_regression]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X_regression, y, test_size=0.2, random_state=42)
    pipeline_regression.fit(X_train, y_train)

    y_pred = pipeline_regression.predict(X_test)
    print("Predicted Prices:", y_pred)

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

    user_df = pd.DataFrame([complete_preferences])
    user_df_transformed = pipeline_clustering['preprocessor'].transform(user_df)
    user_cluster = pipeline_clustering['cluster'].predict(user_df_transformed)[0]

    query = (df['Cluster'] == user_cluster) & (df['price'] <= budget) & (df['city'] == complete_preferences['city'])
    for criterion in strict_criteria:
        if criterion in complete_preferences:
            query &= (df[criterion] == complete_preferences[criterion])

    recommended_houses = df[query].copy()
    recommended_houses['score'] = (recommended_houses['house_type'] == complete_preferences['house_type']).astype(int)
    recommended_houses = recommended_houses.sort_values(by=['score', 'price'], ascending=[False, True]).head(10)

    if recommended_houses.empty:
        return "No houses found that match your preferences within your budget."
    else:
        return recommended_houses[['bedrooms', 'bathrooms', 'house_type', 'city', 'price', 'score']]

# Example usage
df, categorical_cols = load_and_preprocess_data()
predict_prices(df, categorical_cols)
pipeline_clustering = perform_clustering(df, categorical_cols)
user_preferences = {
    'bedrooms': 3, 'bathrooms': 2, 'city': 'London', 'house_type': 'Terraced', 'budget': 700000,
    'strict': ['city', 'bedrooms', 'bathrooms']
}
recommended_houses = recommend_houses(df, user_preferences, pipeline_clustering)
print("Recommended houses based on your preferences:")
print(recommended_houses)

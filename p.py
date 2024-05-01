import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Load the dataset
data_path = 'updated_with_cities.csv'
df = pd.read_csv(data_path)

df.drop(['Unnamed: 0', 'latitude', 'longitude'], axis=1, inplace=True)

# Replace '\N' with NaN
df.replace('\\N', np.nan, inplace=True)

# Drop the 'house_keywords' column if it's not being used
if 'house_keywords' in df.columns:
    df = df.drop('house_keywords', axis=1)

# Define categorical and numerical columns
categorical_cols = ['house_type', 'neighbourhood_quality', 'house_condition', 'crime_rate', 'city']
numerical_cols = ['bedrooms', 'bathrooms', 'garden_size', 'price']


# Convert categorical columns to type 'category' if not already
for col in categorical_cols:
    if df[col].dtype != 'category':
        df[col] = df[col].astype('category')

# Handle missing values if any (optional, based on your data)
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


# Clustering (including price for cluster analysis)
X_clustering = df[categorical_cols + numerical_cols]
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols + ['price'])
    ]
)

# Setup clustering pipeline
pipeline_clustering = Pipeline([
    ('preprocessor', column_transformer),
    ('cluster', KMeans(n_clusters=5, random_state=42))
])
df['Cluster'] = pipeline_clustering.fit_predict(X_clustering)


# Price Prediction (excluding price from features)
X = df.drop(['price'], axis=1)  # Features for prediction
y = df['price']  # Target variable

column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough',  # Apply OneHotEncoder to categorical columns and pass through numerical columns
   
)
column_transformer.set_params(sparse_threshold=0) 

# Setup regression pipeline
pipeline_regression = Pipeline([
    ('preprocessor', column_transformer),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the data types of each column
print(df.dtypes)

# Check for missing values in the DataFrame
print(df.isnull().sum())

# Print unique values for each categorical column
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")


# Convert numeric columns and check for conversion issues
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isnull().any():
        print(f"Non-numeric entries found in {col}")

# Train the regression pipeline
pipeline_regression.fit(X_train, y_train)


#  evaluate the model
y_pred = pipeline_regression.predict(X_test)
print("Predicted Prices:", y_pred)

# Fit the clustering model and assign cluster labels to the DataFrame
df['Cluster'] = pipeline_cluster.fit_predict(X)

# Display the centroids of each cluster to understand common characteristics
centroids = pipeline_cluster.named_steps['cluster'].cluster_centers_
print("Cluster Centers:")
print(centroids)

def recommend_houses(df, user_preferences, pipeline):
    # Define default values for missing preferences
    default_values = {
        'bedrooms': df['bedrooms'].median(),
        'bathrooms': df['bathrooms'].median(),
        'house_type': df['house_type'].mode()[0],
        'city': df['city'].mode()[0],
        'neighbourhood_quality': df['neighbourhood_quality'].mode()[0],
        'house_condition': df['house_condition'].mode()[0],
        'crime_rate': df['crime_rate'].mode()[0],
        'garden_size': df['garden_size'].median(),
        'price': df['price'].median()  # Use median price for internal logic, not exposed to user
    }

    # Merge user preferences with defaults
    complete_preferences = {**default_values, **user_preferences}

    # Extract and remove budget for price filtering
    budget = complete_preferences.pop('budget')

    # Prepare DataFrame for clustering
    user_df = pd.DataFrame([complete_preferences])
    user_df_transformed = pipeline['preprocessor'].transform(user_df)
    user_cluster = pipeline['cluster'].predict(user_df_transformed)[0]

    # Filtering within cluster and under budget
    recommended_houses = df[
        (df['Cluster'] == user_cluster) &
        (df['price'] <= budget) &
        (df['bedrooms'] == complete_preferences['bedrooms']) &
        (df['bathrooms'] == complete_preferences['bathrooms']) &
        # (df['house_type'] == complete_preferences['house_type']) &
        (df['city'] == complete_preferences['city'])
    ].head(10)

    if recommended_houses.empty:
        return "No houses found that match your preferences within your budget."
    else:
        return recommended_houses[['bedrooms', 'bathrooms', 'house_type', 'city', 'price']]

# Example usage
user_preferences = {
    'bedrooms': 3,
    'bathrooms': 2,
    'city': 'London',
    'house_type': 'Terraced',
    'budget': 70000
}

recommended_houses = recommend_houses(df, user_preferences, pipeline_clustering)
print("Recommended houses based on your preferences:")
print(recommended_houses)

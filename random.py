import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('synthetic_house_recommendation_data.csv')

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Assuming 'df' is your DataFrame
# Columns to be one-hot encoded
categorical_columns = ['house_type', 'neighbourhood_quality', 'house_condition', 'crime_rate']

# Set up OneHotEncoder and ColumnTransformer
encoder = OneHotEncoder()
transformer = ColumnTransformer([
    ("one_hot", encoder, categorical_columns)
], remainder='passthrough')

# Apply encoder to the data
df_encoded_array = transformer.fit_transform(df)

# Accessing categories from the fitted transformer
# Get the encoder and then the categories
fitted_encoder = transformer.named_transformers_['one_hot']
categories = fitted_encoder.categories_

# Creating a flat list of new column names for the categorical variables
category_mapping = [f"{cat}__{subcat}" for cat, sublist in zip(categorical_columns, categories) for subcat in sublist]

# Get names of columns that were not transformed (passthrough)
passthrough_indices = [i for i, col in enumerate(df.columns) if col not in categorical_columns]
passthrough_columns = [df.columns[i] for i in passthrough_indices]

# Combine all column names
all_columns = category_mapping + passthrough_columns

# Create the DataFrame with the appropriate column names
df_encoded = pd.DataFrame(df_encoded_array, columns=all_columns)  # Ensure to convert sparse matrix to array

# Check the first few rows of the encoded DataFrame
print(df_encoded.head())

import matplotlib.pyplot as plt
import seaborn as sns

# List of numerical features
numerical_features = ['price', 'bedrooms', 'bathrooms', 'garden_size']

# Sample a subset of the data, say 10% of it
sample_df = df #sample(frac=0.1)

# Plotting distributions of numerical features on the sample
fig, ax = plt.subplots(len(numerical_features), 1, figsize=(8, 4 * len(numerical_features)))
for i, feature in enumerate(numerical_features):
    sns.histplot(sample_df[feature], kde=True, ax=ax[i])
    ax[i].set_title(f'Distribution of {feature}')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Log-transform the 'price' column
# Assuming 'df' is your DataFrame that includes the 'price' column.
df_encoded['log_price'] = np.log1p(df['price'])

# transform the garden size because of extreme right skewness!
# Visualizing the distribution of the log-transformed garden size
df_encoded['garden_size'] = df_encoded['garden_size'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Now apply the log1p transformation
df_encoded['log_garden_size'] = np.log1p(df_encoded['garden_size'])

# Visualize the distribution of the log-transformed garden size
sns.histplot(df_encoded['log_garden_size'], kde=True)
plt.title('Distribution of Log-transformed Garden Size')
plt.xlabel('Log(Garden Size)')
plt.ylabel('Frequency')
plt.show()




# Convert 'log_price' to numeric, coercing errors to NaN
df_encoded['log_price'] = pd.to_numeric(df_encoded['log_price'], errors='coerce')

# Optionally handle NaN values by replacing them with the mean or median
df_encoded['log_price'].fillna(df_encoded['log_price'].mean(), inplace=True)

# List of columns that are known to be binary (one-hot encoded)
binary_columns = [
    'house_type__Bungalow', 'house_type__Detached', 'house_type__Flat', 'house_type__Land', 
    'house_type__Park Home', 'house_type__Semi-Detached', 'house_type__Terraced',
    'neighbourhood_quality__Abysmal', 'neighbourhood_quality__Alright', 
    'neighbourhood_quality__Outstanding', 'neighbourhood_quality__Pleasant', 
    'neighbourhood_quality__Rough', 'house_condition__Abysmal', 'house_condition__Alright', 
    'house_condition__Outstanding', 'house_condition__Pleasant', 'house_condition__Rough', 
    'crime_rate__Extreme', 'crime_rate__High', 'crime_rate__Low', 'crime_rate__Medium', 
    'crime_rate__Non Existent', 'crime_rate__Very High', 'crime_rate__Very Low'
]

# Convert these columns explicitly to integer
for col in binary_columns:
    df_encoded[col] = df_encoded[col].astype(int)

# Verify changes
print(df_encoded[binary_columns].dtypes)

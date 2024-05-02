import pandas as pd

def filter_houses(dataset="updated_with_cities.csv", user_profile, base_preferences):
    """
    Filters houses based on user selections and predefined base preferences.

    Args:
    dataset (pd.DataFrame): The dataset containing house listings.
    user_profile (dict): User choices from the front-end like student/employed, single/married, kids/no kids.
    base_preferences (dict): Base preferences including price range, minimum bedrooms, etc.

    Returns:
    pd.DataFrame: Filtered and sorted list of houses with detailed explanations.
    """
    # Map neighborhood quality to numerical values for sorting
    quality_mapping = {'Outstanding': 5, 'Pleasant': 4, 'Alright': 3, 'Rough': 2, 'Abysmal': 1}
    dataset['quality_score'] = dataset['neighborhood_quality'].map(quality_mapping)

    # Adjust base preferences based on user profile
    if user_profile['employment'] == 'Student':
        base_preferences['max_price'] = min(base_preferences['max_price'], 300000)  # Assuming students prefer cheaper homes
    if user_profile['marital_status'] == 'Married':
        base_preferences['bedrooms'] = max(base_preferences['bedrooms'], 3)  # Assuming married couples might want more space
    if user_profile['children'] == 'Has Kids':
        base_preferences['bedrooms'] = max(base_preferences['bedrooms'], 3)
        base_preferences['neighborhood_quality'] = max(base_preferences['neighborhood_quality'], 4)  # Assuming need for better neighborhoods

    # Filtering logic
    filtered_houses = dataset[
        (dataset['price'] >= base_preferences['min_price']) &
        (dataset['price'] <= base_preferences['max_price']) &
        (dataset['bedrooms'] >= base_preferences['bedrooms']) &
        (dataset['bathrooms'] >= base_preferences['bathrooms']) &
        (dataset['quality_score'] >= quality_mapping[base_preferences['neighborhood_quality']])
    ]

    # Sorting based on neighborhood quality score and price
    filtered_houses = filtered_houses.sort_values(by=['quality_score', 'price'], ascending=[False, True])

    # Generating detailed explanations
    filtered_houses['explanation'] = (
        "Price within budget: $" + filtered_houses['price'].astype(str) + ". " +
        "Has at least " + str(base_preferences['bedrooms']) + " bedrooms and " + 
        str(base_preferences['bathrooms']) + " bathrooms. " +
        "Neighborhood quality is " + filtered_houses['neighborhood_quality'] + ", " +
        "which meets your requirement for " + base_preferences['neighborhood_quality'].lower() + " quality living space."
    )

    return filtered_houses[['price', 'bedrooms', 'bathrooms', 'neighborhood_quality', 'explanation']]

if __name__ == "__main__":
    # Example dataset
    data = {
        'price': [300000, 450000, 250000],
        'bedrooms': [3, 4, 2],
        'bathrooms': [2, 3, 1],
        'neighborhood_quality': [4, 3, 5],
        'crime_rate': [5, 7, 3],
        'house_type': ['Detached', 'Apartment', 'Detached']
    }
    df = pd.DataFrame(data)

    # User preferences
    preferences = {
        'min_price': 200000,
        'max_price': 500000,
        'bedrooms': 2,
        'bathrooms': 2,
        'neighborhood_quality': 3,
        'crime_rate': 6,
        'house_type': ['Detached', 'Apartment']
    }

    # Filter houses
    recommended_houses = filter_houses(df, preferences)
    print(recommended_houses)

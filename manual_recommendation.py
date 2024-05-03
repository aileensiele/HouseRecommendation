import pandas as pd

def filter_houses(user_profile, base_preferences):
    """
    Filters houses based on user selections and predefined base preferences.

    Args:
    df (pd.DataFrame): The df containing house listings.
    user_profile (dict): User choices from the front-end like student/employed, single/married, kids/no kids.
    base_preferences (dict): Base preferences including price range, minimum bedrooms, etc.

    Returns:
    pd.DataFrame: Filtered and sorted list of houses with detailed explanations.
    """
    df = pd.read_csv("updated_with_cities.csv")
    # Map neighborhood quality to numerical values for sorting
    quality_mapping = {'Outstanding': 5, 'Pleasant': 4, 'Alright': 3, 'Rough': 2, 'Abysmal': 1}
    df['quality_score'] = df['neighborhood_quality'].map(quality_mapping)

    # Start explanation
    explanation_parts = []

    # Adjust base preferences based on user profile
    if user_profile['employment'] == 'Student':
        original_max_price = base_preferences['max_price']
        base_preferences['max_price'] = min(base_preferences['max_price'], 300000)  # Assuming students prefer cheaper homes
        explanation_parts.append(f"As a student, we searched for cheaper homes within your budget of ${base_preferences['max_price']:,.2f}, reduced from ${original_max_price:,.2f}.")

    if user_profile['marital_status'] == 'Married':
        explanation_parts.append("Because you are married, we recommend homes with at least 3 bedrooms to ensure ample space.")

    if user_profile['children'] == 'Has Kids':
        explanation_parts.append("Since you have kids, we narrowed the search to neighborhoods rated 'Pleasant' or better to ensure a safer and more suitable living environment.")

    # Apply filters
    filtered_houses = df[
        (df['price'] >= base_preferences['min_price']) &
        (df['price'] <= base_preferences['max_price']) &
        (df['bedrooms'] >= base_preferences['bedrooms']) &
        (df['bathrooms'] >= base_preferences['bathrooms']) &
        (df['quality_score'] >= quality_mapping.get(base_preferences['neighborhood_quality'], 3))
    ]

    # Sorting based on neighborhood quality score and price
    filtered_houses = filtered_houses.sort_values(by=['quality_score', 'price'], ascending=[False, True])

    # Generate detailed explanations combining specific attributes
    filtered_houses['explanation'] = (
        "Price within budget: $" + filtered_houses['price'].astype(str) + ". " +
        "Has at least " + str(base_preferences['bedrooms']) + " bedrooms and " +
        str(base_preferences['bathrooms']) + " bathrooms. " +
        "Neighborhood quality is " + filtered_houses['neighborhood_quality'] + ". " +
        " ".join(explanation_parts)
    )

    return filtered_houses[['price', 'bedrooms', 'bathrooms', 'neighborhood_quality', 'explanation']]

# Example usage of the function
# Define a df (mock or actual DataFrame), user profile, and base preferences before using this function.

if __name__ == "__main__":
    # User profiles
    student_profile = {
        'employment': 'Student',
        'marital_status': 'Single',
        'children': 'No Kids'
    }

    married_with_kids_profile = {
        'employment': 'Employed',
        'marital_status': 'Married',
        'children': 'Has Kids'
    }

    # Base preferences for both profiles
    base_preferences_student = {
        'min_price': 100000,
        'max_price': 400000,
        'bedrooms': 2,
        'bathrooms': 1,
        'neighborhood_quality': 'Alright'
    }

    base_preferences_married = {
        'min_price': 200000,
        'max_price': 500000,
        'bedrooms': 3,
        'bathrooms': 2,
        'neighborhood_quality': 'Pleasant'
    }

    # Testing the function for a student
    recommended_houses_student = filter_houses(student_profile, base_preferences_student)
    print("Recommendations for Student:")
    print(recommended_houses_student)

    # Testing the function for a married person with kids
    recommended_houses_married = filter_houses(married_with_kids_profile, base_preferences_married)
    print("Recommendations for Married with Kids:")
    print(recommended_houses_married)
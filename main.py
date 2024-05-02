from flask import Flask, request, render_template, jsonify
import pandas as pd
# import functs from decision to apply in this model!
from decision import load_and_preprocess_data, perform_clustering, recommend_houses
from linear_regression import train_and_predict_price


app = Flask(__name__)

# Load the cities list when the application starts
cities = []
def load_cities():
    global cities
    df = pd.read_csv('updated_with_cities.csv')  # Load cities from CSV
    cities = df['city'].unique().tolist()  # Update the global variable
load_cities()

# Get these things done with before proceeding
# Global variables for the model and data ( clustering algo)
df, categorical_cols = load_and_preprocess_data()
pipeline_clustering = perform_clustering(df, categorical_cols)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quick_estimate', methods=['GET', 'POST'])
def quick_estimate():
    if request.method == 'POST':
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        city = request.form.get('city')
        estimated_price, explanation = train_and_predict_price(bedrooms, bathrooms, city)  # Ensure df is available here
        
        if estimated_price is None:
            return jsonify({'error': explanation})
        return jsonify({'estimated_price': f"${estimated_price:,.2f}", 'explanation': explanation})
    return render_template('quick_price.html', cities=cities)

@app.route('/ai_recommender', methods=['GET', 'POST'])
def ai_recommender():
    if request.method == 'POST':
        form_data = request.form
        print(request.form)
        recommended_properties = get_ai_recommendations(form_data)
    
        return jsonify(recommended_properties)
    return render_template('advanced_ai.html', cities=cities)  # Ensure you have advanced_ai.html

@app.route('/manual_recommend', methods=['GET', 'POST'])
def manual_recommend():
    if request.method == 'POST':
        form_data = request.form
        recommendations = get_manual_recommendations(form_data)
        return jsonify(recommendations)
    return render_template('personal_recommendation.html', cities=cities)  # Ensure you have personal_recommendation.html

# def calculate_quick_estimate(form_data):
#     # replace with calling Dan's linear regression function
#     user_preferences = {
#         'bedrooms': int(form_data['bedrooms']),
#         'bathrooms': int(form_data['bathrooms']),
#         'city': form_data['city'],
#         }
#     predicted_price, explanation = train_and_predict_price()
#     return {'estimate': predicted_price,
#             'explanation': explanation
#         }

def get_ai_recommendations(form_data):
    # replace with calling decision tree module
        # Convert form data to appropriate types and prepare for model input
    user_preferences = {
        'bedrooms': int(form_data['bedrooms']),
        'bathrooms': int(form_data['bathrooms']),
        'city': form_data['city'],
        'house_type': form_data['house_type'],
        'house_condition': form_data['house_condition'],
        'neighbourhood_quality': form_data['neighborhood_quality'],  # Note spelling difference in key
        'crime_rate': form_data['crime_rate'],
        'garden_size': float(form_data['garden_size']),
        'budget': float(form_data['budget'])
    }

    # Call the model prediction function
    recommended_houses, explanation = recommend_houses(df, user_preferences, pipeline_clustering)
    
    if isinstance(recommended_houses, str):  # If no houses were found
        return {'result': recommended_houses}
    else:
        # Format the results as a dictionary for JSON conversion
        results = recommended_houses.to_dict(orient='records')
        print(results)
        return {
            'recommended_properties': results,
            'explanation': explanation
        }

def get_manual_recommendations(data):
    # replace with calling manual rec funct
    return {'result': 'Manually curated recommendations based on rules'}

if __name__ == '__main__':
    app.run(debug=True)

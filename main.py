from flask import Flask, request, render_template, jsonify
import pandas as pd
# import functs from decision to apply in this model!
from decision import load_and_preprocess_data, perform_clustering, recommend_houses
from linear_regression import train_and_predict_price
from manual_recommendation import filter_houses


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
    return render_template('advanced_ai.html', cities=cities)  

@app.route('/manual_recommend', methods=['GET', 'POST'])
def manual_recommend():
    if request.method == 'POST':
        # Collect form data
        user_profile = {
            'employment': request.form.get('employment'), 
            'marital_status': request.form.get('marital_status'),  
            'children': request.form.get('children')  
        }
        base_preferences = {
            'min_price': float(request.form.get('min_price')),
            'max_price': float(request.form.get('max_price')),
            'bedrooms': int(request.form.get('bedrooms')),
            'bathrooms': int(request.form.get('bathrooms')),
            'neighborhood_quality': request.form.get('neighborhood_quality')
        }
        
        # Load dataset and filter houses based on user preferences
        df = pd.read_csv('updated_with_cities.csv')  # Load the dataset if not already loaded
        filtered_houses = filter_houses(user_profile, base_preferences)
        
        # Convert the results to a JSON-friendly format if not empty
        if not filtered_houses.empty:
            results = filtered_houses.to_dict(orient='records')
            print(results)
            return jsonify({'result': results})
        else:
            return jsonify({'error': 'No matching houses found.'})

    # Load initial page
    return render_template('personal_recommendation.html', cities=cities)


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

if __name__ == '__main__':
    app.run(debug=True)

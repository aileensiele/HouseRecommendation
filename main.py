from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from random_forest import preprocess_data, train_model, predict_price  # Assuming these functions are in random_forest.py

app = Flask(__name__)

@app.route('/')
def index():
    # Make sure the HTML file is in the 'templates' directory
    return render_template('main.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Extract data from form, using .get to avoid KeyError if keys don't exist
        bedrooms = int(request.form.get('bedrooms', 0))
        bathrooms = int(request.form.get('bathrooms', 0))
        city = request.form.get('city', '')
        house_type = request.form.get('house_type', '')
        house_condition = request.form.get('house_condition', '')
        neighborhood_quality = request.form.get('neighborhood_quality', '')
        crime_rate = request.form.get('crime_rate', '')
        garden_size = float(request.form.get('garden_size', 0))
        budget = float(request.form.get('budget', 0))

        # Create a dictionary with the input data
        user_input = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'city': city,
            'house_type': house_type,
            'house_condition': house_condition,
            'neighborhood_quality': neighborhood_quality,
            'crime_rate': crime_rate,
            'garden_size': garden_size,
            'budget': budget
        }

        # Preprocess and predict
        processed_input = preprocess_data(pd.DataFrame([user_input]))
        # Check what you are getting
        print(processed_input)

        # Depending on button pressed, call different functionality
        action = request.form.get('action', 'estimate')  # Default to 'estimate' if not found
        if action == 'estimate':
            predicted_price = predict_price(processed_input)
            return jsonify({'predicted_price': f"${predicted_price:.2f}"})
        elif action == 'recommend':
            recommendations = recommend_houses(processed_input)  # Implement this
            return jsonify(recommendations)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})
    
    @app.route('/quick_estimate', methods=['POST'])
	def quick_estimate():
		# Extract and process data for quick price estimation
		bedrooms = request.form['bedrooms']
		bathrooms = request.form['bathrooms']
		location = request.form['location']
		
		price = calculate_quick_estimate(bedrooms, bathrooms, location)
		# explanation eg Given these number of bedrooms, bathrooms, and location, we have used a linear regression model to estimate ___ price. Because _____
		return render_template('results.html', price=price, type='Quick Estimate')

	@app.route('/ai_recommender', methods=['POST'])
	def ai_recommender():
		# Process data for comprehensive AI-based recommendation
		# Similar to quick_estimate, but more detailed
		# will use decision tree here, to estimate using advanced features, which are ___ what the user entred
		return render_template('ai_recommendations.html')

	@app.route('/manual_recommend', methods=['POST'])
	def manual_recommend():
		# Manually process data based on if-else conditions
		# since you desire a near personalized recommendation
		return render_template('manual_recommendation.html')

if __name__ == '__main__':
    app.run(debug=True)

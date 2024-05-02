from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# Load the cities list when the application starts
cities = []
def load_cities():
    global cities
    df = pd.read_csv('updated_with_cities.csv')  # Load cities from CSV
    cities = df['city'].unique().tolist()  # Update the global variable
load_cities()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quick_estimate', methods=['GET', 'POST'])
def quick_estimate():
    if request.method == 'POST':
        bedrooms = request.form.get('bedrooms')
        bathrooms = request.form.get('bathrooms')
        city = request.form.get('city')
        estimated_price = calculate_quick_estimate(bedrooms, bathrooms, city)
        return jsonify({'estimated_price': f"${estimated_price:.2f}"})
    return render_template('quick_price.html', cities=cities)  # Ensure you have quick_price.html

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

def calculate_quick_estimate(bedrooms, bathrooms, city):
    # replace with calling Dan's linear regression function
    return 100000 + int(bedrooms) * 50000 + int(bathrooms) * 30000

def get_ai_recommendations(data):
    # replace with calling decision tree module
    return {'result': 'AI-based recommendations based on the provided data'}

def get_manual_recommendations(data):
    # replace with calling manual rec funct
    return {'result': 'Manually curated recommendations based on rules'}

if __name__ == '__main__':
    app.run(debug=True)


# House Recommendation and Price Estimation System

## Overview
HouseRecommendation is an advanced analytical tool designed to streamline the home-buying process by leveraging cutting-edge machine learning algorithms. This system offers precise house price estimations and personalized recommendations by integrating Linear Regression, Decision Trees, and Random Forests. Aimed at enhancing user satisfaction, HouseRecommendation simplifies the decision-making process for prospective homebuyers by aligning house options with their preferences and budget constraints.

### Features
- **Price Estimation with Linear Regression:** Our system employs Linear Regression to provide a predictive model that estimates house prices with high accuracy. This model specifically analyzes the influence of the number of bathrooms and bedrooms on house prices, allowing users to understand price determinants effectively.

- **Comprehensive Price Estimation using Decision Trees:** The Decision Tree algorithm is utilized to analyze all available house features to estimate costs accurately. This method facilitates a thorough evaluation of properties, ensuring that all relevant characteristics such as location, neighboorhood quality, and house condition are considered.

- **Advanced Clustering for Customized Recommendations:** Our model uses the clustering mechanism (KMeans) that groups the houses into clusters based on their features including 'price'. This model is trained to find patterns or groups of similar houses in the dataset.Through the clustering method, our system groups houses into distinct categories based on features specified by users. This approach tailors the search to user preferences and prioritizes these clusters according to the user's budget as well as any other preferences that they indicate as they input desired features.

- **User-Centric Scoring System:** After identifying clusters, our system assigns a score to each house based on its congruence with the user's specified needs. This scoring mechanism prioritizes homes that best match the user’s preferences, offering a ranked list of recommendations that maximize satisfaction.

## Challenges Faced
Our journey in developing the HouseRecommendation system included several significant challenges:
- Harmonizing diverse machine learning models to ensure seamless operation and integration.
- Achieving high accuracy in predictions while catering to various user preferences and diverse property types.
- Developing an intuitive user interface that effectively captures complex user inputs and translates them into actionable outputs.

## How to Run the Program
To utilize the HouseRecommendation system, please follow the steps outlined below:

1. **Prerequisites:**
   - Install Python (version 3.8 or newer recommended) on your machine.
   - Install all required Python libraries using the following command:
     ```
     pip install numpy pandas scikit-learn matplotlib seaborn
     ```

2. **Clone the Repository:**
   - Get a copy of the source code on your local machine by executing:
     ```
     git clone https://github.com/aileensiele/HouseRecommendation.git
     ```

3. **Execute the Script:**
   - Navigate to the directory containing the cloned repository and start the application:
     ```
     cd HouseRecommendation
     python main.py
     ```

4. **Interact with the Application:**
   - Enter your housing preferences as prompted on the html page to receive tailored house recommendations.


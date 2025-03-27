from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and train models
def load_models():
    # Simple Linear Regression
    simple_data = pd.read_csv('datasets/house_price_simple.csv')
    X_simple = simple_data[['area']]
    y_simple = simple_data['price']
    simple_linear_model = LinearRegression()
    simple_linear_model.fit(X_simple, y_simple)

    # Multiple Linear Regression
    multiple_data = pd.read_csv('datasets/house_price_multiple.csv')
    X_multiple = multiple_data[['area', 'bedrooms', 'bathrooms', 'distance_to_city', 'parking']]
    y_multiple = multiple_data['price']
    multiple_linear_model = LinearRegression()
    multiple_linear_model.fit(X_multiple, y_multiple)

    # Polynomial Regression
    temp_data = pd.read_csv('datasets/temperature.csv')
    X_poly = temp_data[['hour']]
    y_poly = temp_data['temperature']
    poly_features = PolynomialFeatures(degree=2)
    X_poly_transformed = poly_features.fit_transform(X_poly)
    poly_model = LinearRegression()
    poly_model.fit(X_poly_transformed, y_poly)

    # KNN Classification for Iris
    iris_data = pd.read_csv('datasets/iris.csv')
    X_iris = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_iris = iris_data['species']
    label_encoder = LabelEncoder()
    y_iris_encoded = label_encoder.fit_transform(y_iris)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_iris, y_iris_encoded)

    # Logistic Regression
    pass_data = pd.read_csv('datasets/student_pass.csv')
    X_logistic = pass_data[['study_hours', 'previous_score']]
    y_logistic = pass_data['passed']
    logistic_model = LogisticRegression()
    logistic_model.fit(X_logistic, y_logistic)

    return {
        'simple_linear': simple_linear_model,
        'multiple_linear': multiple_linear_model,
        'polynomial': (poly_model, poly_features),
        'knn': (knn_model, label_encoder),
        'logistic': logistic_model
    }

models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simple-linear')
def simple_linear():
    return render_template('simple_linear.html')

@app.route('/multiple-linear')
def multiple_linear():
    return render_template('multiple_linear.html')

@app.route('/polynomial')
def polynomial():
    return render_template('polynomial.html')

@app.route('/knn')
def knn_page():
    return render_template('knn.html')

@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data['model_type']
    
    try:
        if model_type == 'simple_linear':
            area = float(data['area'])
            prediction = models['simple_linear'].predict([[area]])[0]
            return jsonify({'prediction': f'${prediction:,.2f}'})

        elif model_type == 'multiple_linear':
            features = [
                float(data['area']),
                float(data['bedrooms']),
                float(data['bathrooms']),
                float(data['distance']),
                float(data['parking'])
            ]
            prediction = models['multiple_linear'].predict([features])[0]
            return jsonify({'prediction': f'${prediction:,.2f}'})

        elif model_type == 'polynomial':
            hour = float(data['hour'])
            poly_model, poly_features = models['polynomial']
            prediction = poly_model.predict(poly_features.transform([[hour]]))[0]
            return jsonify({'prediction': f'{prediction:.1f}°C'})

        elif model_type == 'knn':
            # Get KNN model and label encoder
            knn_model, label_encoder = models['knn']
            
            # Get features from request
            features = [
                float(data['sepal_length']),
                float(data['sepal_width']),
                float(data['petal_length']),
                float(data['petal_width'])
            ]
            
            # Make prediction
            prediction = knn_model.predict([features])[0]
            probabilities = knn_model.predict_proba([features])[0]
            
            # Get the predicted species name and confidence
            species = label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            return jsonify({
                'species': species,
                'confidence': confidence
            })

        elif model_type == 'logistic':
            study_hours = float(data['study_hours'])
            previous_score = float(data['previous_score'])
            features = [[study_hours, previous_score]]
            
            # Get prediction and probability
            prediction = models['logistic'].predict(features)[0]
            probability = models['logistic'].predict_proba(features)[0]
            
            # Return the probability of the predicted class
            confidence = probability[1] if prediction == 1 else probability[0]
            
            return jsonify({
                'prediction': int(prediction),
                'probability': float(confidence)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

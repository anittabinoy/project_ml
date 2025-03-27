import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

def evaluate_simple_linear():
    print("\n=== Simple Linear Regression ===")
    data = pd.read_csv('datasets/house_price_simple.csv')
    X = data[['area']]
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.title('Simple Linear Regression: House Price Prediction')
    plt.legend()
    plt.savefig('static/simple_linear_plot.png')
    plt.close()

def evaluate_multiple_linear():
    print("\n=== Multiple Linear Regression ===")
    data = pd.read_csv('datasets/house_price_multiple.csv')
    X = data[['area', 'bedrooms', 'bathrooms', 'distance_to_city', 'parking']]
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print("\nFeature Importance:")
    for feature, importance in zip(X.columns, model.coef_):
        print(f"{feature}: {importance:,.2f}")

def evaluate_polynomial():
    print("\n=== Polynomial Regression ===")
    data = pd.read_csv('datasets/temperature.csv')
    X = data[['hour']]
    y = data['temperature']
    
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}°C")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    X_sorted = np.sort(X.values, axis=0)
    X_poly_sorted = poly_features.transform(X_sorted)
    y_pred_sorted = model.predict(X_poly_sorted)
    
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X_sorted, y_pred_sorted, color='red', label='Predicted')
    plt.xlabel('Hour')
    plt.ylabel('Temperature (°C)')
    plt.title('Polynomial Regression: Temperature Prediction')
    plt.legend()
    plt.savefig('static/polynomial_plot.png')
    plt.close()

def evaluate_knn():
    print("\n=== KNN Regression ===")
    data = pd.read_csv('datasets/student_scores.csv')
    X = data[['study_hours', 'sleep_hours']]
    y = data['score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} points")

def evaluate_logistic():
    print("\n=== Logistic Regression ===")
    data = pd.read_csv('datasets/admission.csv')
    X = data[['exam1_score', 'exam2_score']]
    y = data['admitted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plotting decision boundary
    plt.figure(figsize=(10, 6))
    x_min, x_max = X['exam1_score'].min() - 1, X['exam1_score'].max() + 1
    y_min, y_max = X['exam2_score'].min() - 1, X['exam2_score'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y == 0]['exam1_score'], X[y == 0]['exam2_score'], color='red', label='Not Admitted')
    plt.scatter(X[y == 1]['exam1_score'], X[y == 1]['exam2_score'], color='blue', label='Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Logistic Regression: Admission Decision Boundary')
    plt.legend()
    plt.savefig('static/logistic_plot.png')
    plt.close()

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    import os
    if not os.path.exists('static'):
        os.makedirs('static')
        
    print("Evaluating all models...")
    evaluate_simple_linear()
    evaluate_multiple_linear()
    evaluate_polynomial()
    evaluate_knn()
    evaluate_logistic()
    print("\nEvaluation complete. Visualization plots have been saved in the 'static' directory.")

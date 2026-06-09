# CodeAlpha_Car-Price-Prediction-
## Overview

This project predicts the selling price of used cars using Machine Learning techniques. Various regression algorithms were trained and evaluated to estimate a car's selling price based on factors such as present price, age, kilometers driven, fuel type, transmission type, seller type, and ownership history.

A Streamlit web application was also developed to provide real-time price predictions through a user-friendly interface.

---

## Objective

The objective of this project is to build a regression model capable of accurately predicting the selling price of used cars using historical vehicle data and machine learning algorithms.

---

## Dataset

The dataset contains information about used cars and their selling prices.

### Features

| Feature       | Description                                |
| ------------- | ------------------------------------------ |
| Car_Name      | Name of the car model                      |
| Year          | Manufacturing year                         |
| Present_Price | Current showroom price (Lakhs)             |
| Driven_kms    | Total kilometers driven                    |
| Fuel_Type     | Petrol, Diesel, or CNG                     |
| Selling_type  | Dealer or Individual                       |
| Transmission  | Manual or Automatic                        |
| Owner         | Number of previous owners                  |
| Selling_Price | Selling price of the car (Target Variable) |

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed duplicate records
* Checked for missing values
* Created a new feature: `Car_Age = 2026 - Year`
* Dropped unnecessary columns:

  * Car_Name
  * Year
* Applied One-Hot Encoding to categorical features
* Scaled numerical features using StandardScaler
* Built a preprocessing pipeline using ColumnTransformer and Pipeline

---

## Exploratory Data Analysis

The dataset was analyzed using:

* Descriptive statistics
* Correlation matrix
* Correlation heatmap
* Feature importance analysis

### Key Findings

* Present Price has the strongest positive relationship with Selling Price.
* Car Age negatively impacts resale value.
* Driven Kilometers have a relatively smaller impact.

---

## Feature Engineering

A new feature was created:

```python
Car_Age = 2026 - Year
```

This feature helps capture vehicle depreciation over time and improves prediction performance.

---

## Machine Learning Models Evaluated

The following regression algorithms were trained and compared:

### Linear Regression

* Train R²: 0.90
* Test R²: 0.75
* Cross Validation R²: 0.85

### Random Forest Regressor

* Train R²: 0.98
* Test R²: 0.56
* Cross Validation R²: 0.86

### Tuned Random Forest Regressor

* Train R²: 0.98
* Test R²: 0.51
* Cross Validation R²: 0.84

### Decision Tree Regressor

* Train R²: 1.00
* Test R²: 0.83
* Cross Validation R²: 0.88

### Tuned Decision Tree Regressor

* Train R²: 0.98
* Test R²: 0.82
* Cross Validation R²: 0.90

### LightGBM Regressor

* Train R²: 0.90
* Test R²: 0.68
* Cross Validation R²: 0.80

---

## Model Evaluation Metrics

### Final Selected Model: Decision Tree Regressor

| Metric   | Value |
| -------- | ----- |
| R² Score | 0.83  |
| MAE      | 1.16  |
| RMSE     | 2.17  |

---

## Feature Importance Analysis

Feature importance from the Decision Tree model:

| Feature                 | Importance |
| ----------------------- | ---------- |
| Present_Price           | 0.927553   |
| Car_Age                 | 0.070450   |
| Driven_kms              | 0.001997   |
| Owner                   | 0.000000   |
| Fuel_Type_Diesel        | 0.000000   |
| Fuel_Type_Petrol        | 0.000000   |
| Selling_type_Individual | 0.000000   |
| Transmission_Manual     | 0.000000   |

### Observation

Present Price is the most influential factor in determining the selling price of a vehicle. Car Age also contributes to the prediction, while other features have comparatively smaller effects.

---

## Visualizations

The following plots were generated:

* Correlation Heatmap(images/correlationmap.png)
* Feature Importance Plot(images/Feature Importance.png)
* Residual Plot(images/Resiudal Plot.png)
* Actual vs Predicted Prices Plot(images/Actual vs Predicted Prices.png)

These visualizations helped evaluate model behavior and identify important predictors.

---

## Streamlit Web Application

A Streamlit-based web application was developed to provide real-time car price predictions.

### User Inputs

* Present Price
* Driven Kilometers
* Car Age
* Fuel Type
* Seller Type
* Transmission Type

### Output

* Estimated Selling Price (in Lakhs)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* LightGBM
* Joblib
* Streamlit

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Model Saving
10. Streamlit Deployment

---

## Conclusion

A machine learning system was successfully developed to predict used car selling prices. Multiple regression algorithms were evaluated and compared. The Decision Tree model achieved the best predictive performance on the test data, while Linear Regression demonstrated strong generalization and interpretability.

The analysis revealed that Present Price and Car Age are the most significant factors affecting resale value. The final model was integrated into a Streamlit web application, allowing users to obtain instant price predictions.

---



---

## Author

Machine Learning Project – Car Price Prediction

Developed using Python, Scikit-Learn, and Streamlit.

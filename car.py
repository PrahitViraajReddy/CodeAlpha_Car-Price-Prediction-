import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import joblib
try:
    df=pd.read_csv("car data.csv")
except FileNotFoundError:
    exit(0)

print(df.head())
print(df.tail())
print(df.describe())
print(df.columns)
print(df.dtypes)
df = df.drop_duplicates()
print(df.duplicated().sum())
print(df.isnull().sum())
corr=df.select_dtypes(include=np.number).corr()
print(corr["Selling_Price"].sort_values(ascending=False))
sns.heatmap(
    corr,
    annot=True,      # show correlation values
    cmap='coolwarm', # color scheme
    fmt='.2f',       # 2 decimal places
    linewidths=0.5   # cell borders
)
plt.show()

df["Car_Age"] = 2026 - df["Year"]
df.drop("Year", axis=1, inplace=True)
df.drop("Car_Name", axis=1, inplace=True)
print(df.head())

X=df.drop(["Selling_Price"],axis=1)
y=df["Selling_Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# column groups
numeric_features = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42,
    max_depth=5))
])
model.fit(X_train,y_train)

joblib.dump(model, "car_price_model.pkl")
pred=model.predict(X_test)
tpred=model.predict(X_train)
print(pred)
print(tpred)
score=r2_score(y_test,pred)
tscore=r2_score(y_train,tpred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("MAE:", mae)
print("RMSE:", rmse)
print(score)
print(tscore)
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

coefficients = model.named_steps["regressor"].feature_importances_

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": np.abs(coefficients)
}).sort_values(by="Importance", ascending=False)

print(importance)
plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance (Absolute Coefficient)")
plt.title("Feature Importance - Linear Regression")
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring='r2'
)
print(scores)
print(scores.mean())
residuals = y_test - pred

plt.figure(figsize=(8,5))
plt.scatter(pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
y_test_arr = np.array(y_test)
pred_arr = np.array(pred)

# sort by actual values
sorted_idx = np.argsort(y_test_arr)

plt.figure(figsize=(10,6))

plt.plot(
    y_test_arr[sorted_idx],
    label="Actual Price",
    marker='o'
)

plt.plot(
    pred_arr[sorted_idx],
    label="Predicted Price",
    marker='x'
)

plt.xlabel("Sorted Samples")
plt.ylabel("Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.show()


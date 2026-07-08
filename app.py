import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("🏡 House Price Prediction App with City")

# Sample dataset (demo)
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Age": [10, 5, 8, 2, 1],
    "City": ["Lucknow", "Delhi", "Mumbai", "Lucknow", "Delhi"],
    "Price": [50, 70, 90, 120, 150]  # in lakhs
}
df = pd.DataFrame(data)

# Encode city (convert text to numbers)
df["City_Code"] = df["City"].astype("category").cat.codes

# Features and target
X = df[["Area", "Bedrooms", "Age", "City_Code"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Enter House Details")
area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
age = st.sidebar.number_input("Age of House (years)", min_value=0, max_value=50, value=5)

# City dropdown
city = st.sidebar.selectbox("City", df["City"].unique())
city_code = df["City"].astype("category").cat.categories.get_loc(city)

# Prediction
input_data = np.array([[area, bedrooms, age, city_code]])
predicted_price = model.predict(input_data)[0]

# Output
st.subheader("Predicted House Price")
st.write(f"💰 Estimated Price in {city}: {predicted_price:.2f} lakhs")

# Show dataset
st.subheader("Training Data")
st.dataframe(df)


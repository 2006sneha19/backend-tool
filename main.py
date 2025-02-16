from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# ✅ Database Connection with SQLAlchemy
DB_URL = "postgresql://postgres:Naruto%4018%232@localhost/Nutrition_ai"
engine = create_engine(DB_URL)

# ✅ Define Meal Categories
meal_types = {
    "Breakfast": ["paratha", "idli", "poha", "dosa", "pancake", "bread", "omelette", "cereal", "chai", "tea", "coffee"],
    "Lunch": ["biryani", "curry", "dal", "rice", "roti", "naan", "fish", "chicken", "mutton", "salad", "stir-fry"],
    "Dinner": ["pasta", "stew", "soup", "salad", "grilled", "noodles", "wrap", "omelette", "khichdi", "pulao"]
}

# ✅ Load Data from PostgreSQL
def get_food_data():
    try:
        query = "SELECT food_name, energy_kcal, protein_g, carb_g, fat_g FROM nutrition_csv_data;"
        df = pd.read_sql(query, engine)
        return df.dropna()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ✅ Prepare Data for AI Model
df = get_food_data()

X = df[['energy_kcal', 'protein_g', 'carb_g', 'fat_g']].values
y = np.arange(len(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split Data (80% Training, 20% Validation)
split_index = int(0.8 * len(X_scaled))
X_train, X_val = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# ✅ Load and compile the model at app startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, verbose=1)

# ✅ FastAPI Routes

@app.get("/")
def home():
    return {"message": "AI NUTRITION ASSISTANT BACKEND"}

@app.post("/recommend_meals")
def recommend_meals(user_data: dict):
    try:
        dietary_preference = user_data.get("dietary_preference", "").lower()
        allergies = user_data.get("allergies", "none").lower()

        # ✅ Filter Foods Based on Dietary Preference
        filtered_df = df.copy()
        if dietary_preference == "veg":
            filtered_df = filtered_df[~filtered_df['food_name'].str.contains("chicken|fish|meat|egg|mutton|prawn|beef|seafood", case=False)]
        elif dietary_preference == "vegan":
            filtered_df = filtered_df[~filtered_df['food_name'].str.contains("chicken|fish|meat|egg|milk|cheese|butter|mutton|prawn|beef|seafood", case=False)]
        elif dietary_preference == "non-veg":
            filtered_df = filtered_df[filtered_df['food_name'].str.contains("chicken|fish|meat|egg|mutton|seafood|prawn|lamb|beef", case=False)]

        # ✅ Remove Allergy Ingredients
        if allergies != "none":
            allergy_list = allergies.split(",")
            for allergy in allergy_list:
                filtered_df = filtered_df[~filtered_df['food_name'].str.contains(allergy.strip(), case=False)]

        # ✅ Recommendation Logic
        def recommend(meal_type, macros):
            meals = filtered_df[filtered_df['food_name'].str.contains("|".join(meal_types[meal_type]), case=False, na=False)]
            if meals.empty:
                return ["No suitable meals found"]
            sample_input_scaled = scaler.transform([macros])
            predicted_indices = np.argsort(model.predict(sample_input_scaled).flatten())[:10]
            return list(meals.iloc[predicted_indices]['food_name'].unique())[:3]

        return {
            "breakfast": recommend("Breakfast", [400, 15, 50, 10]),
            "lunch": recommend("Lunch", [700, 30, 80, 20]),
            "dinner": recommend("Dinner", [600, 25, 70, 15])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/predict")
def predict():
    return {"prediction": "Your AI result here"}

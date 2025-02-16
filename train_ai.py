import psycopg2
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Connect to PostgreSQL
def get_food_data():
    conn = psycopg2.connect(
        dbname="Nutrition_ai",
        user="postgres",
        password="Naruto@18#2",
        host="localhost",
        port="5432"
    )
    query = "SELECT food_name, energy_kcal, protein_g, carb_g, fat_g FROM nutrition_csv_data;"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Handling missing values explicitly
    df.fillna({"food_name": "Unknown", "energy_kcal": 0, "protein_g": 0, "carb_g": 0, "fat_g": 0}, inplace=True)
    return df

# Load and preprocess data
df = get_food_data()
X = df[['energy_kcal', 'protein_g', 'carb_g', 'fat_g']].values
y = np.arange(len(X))  # Placeholder labels (indices)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# AI Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Train model on 80% training data
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val), verbose=1)

# Meal categories
meal_types = {
    "Breakfast": ["paratha", "idli", "poha", "dosa", "pancake", "bread", "omelette", "cereal", "chai", "tea", "coffee"],
    "Lunch": ["biryani", "curry", "dal", "rice", "roti", "naan", "fish", "chicken", "mutton", "salad", "stir-fry"],
    "Dinner": ["pasta", "stew", "soup", "salad", "grilled", "noodles", "wrap", "omelette", "khichdi", "pulao"]
}

# User Input
print("\n🔹 Enter Your Details for Personalized Nutrition Plan 🔹")
age = int(input("Enter your age: "))
weight = float(input("Enter your weight (kg): "))
height = float(input("Enter your height (cm): "))
gender = input("Enter your gender (Male/Female): ").strip().lower()
health_conditions = input("Any health conditions? (e.g., Diabetes, High BP, None): ").strip().lower()
dietary_preference = input("Dietary preference (Veg, Non-Veg, Vegan, Keto, etc.): ").strip().lower()
activity_level = input("Activity level (Sedentary, Moderate, Active): ").strip().lower()
allergies = input("Any food allergies? (e.g., Peanuts, Dairy, None): ").strip().lower()
daily_calorie_goal = float(input("Enter your daily calorie goal: "))

# Filtering foods
filtered_df = df.copy()

if dietary_preference == "veg":
    filtered_df = filtered_df[~filtered_df['food_name'].str.contains("chicken|fish|meat|egg|mutton|prawn|beef|seafood", case=False)]
elif dietary_preference == "vegan":
    filtered_df = filtered_df[~filtered_df['food_name'].str.contains("chicken|fish|meat|egg|milk|cheese|butter|mutton|prawn|beef|seafood", case=False)]
elif dietary_preference == "non-veg":
    filtered_df = filtered_df[filtered_df['food_name'].str.contains("chicken|fish|meat|egg|mutton|seafood|prawn|lamb|beef", case=False)]

if allergies != "none":
    allergy_list = allergies.split(",")
    for allergy in allergy_list:
        filtered_df = filtered_df[~filtered_df['food_name'].str.contains(allergy.strip(), case=False)]

# Function to recommend diverse meals
def recommend_meals(meal_type, macros):
    filtered_meals = filtered_df[filtered_df['food_name'].str.contains("|".join(meal_types[meal_type]), case=False, na=False)]
    if filtered_meals.empty:
        return ["No suitable meals found"]
    
    sample_input_scaled = scaler.transform([macros])
    predicted_indices = np.argsort(model.predict(sample_input_scaled).flatten())[:10]
    
    selected_meals = list(filtered_meals.iloc[predicted_indices]['food_name'].unique())[:3]
    while len(selected_meals) < 3:
        selected_meals.append(filtered_meals.sample(1)['food_name'].values[0])
    
    return selected_meals

# Macronutrient targets
breakfast_macros = [400, 15, 50, 10]
lunch_macros = [700, 30, 80, 20]
dinner_macros = [600, 25, 70, 15]

# Generate meal plan
breakfast = recommend_meals("Breakfast", breakfast_macros)
lunch = recommend_meals("Lunch", lunch_macros)
dinner = recommend_meals("Dinner", dinner_macros)

# Display meal plan
print("\n🍽️ AI Recommended Meal Plan:")
print(f"✅ Breakfast: {', '.join(breakfast)}")
print(f"✅ Lunch: {', '.join(lunch)}")
print(f"✅ Dinner: {', '.join(dinner)}")
print("\n🔥 Stay Healthy & Enjoy Your Meals!")

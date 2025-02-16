import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="Nutrition_ai",
    user="postgres",
    password="Naruto@18#2",  # Replace with your actual password
    host="localhost",
    port="5432"
)

# Read the CSV file
df = pd.read_csv("C:\\Users\\f\\Downloads\\indian_food_cleaned.csv")  # Change this to your correct file path

# Insert data into PostgreSQL
cur = conn.cursor()
for index, row in df.iterrows():
    cur.execute(
        "INSERT INTO nutrition_csv_data (food_code, food_name, energy_kj, energy_kcal, carb_g, protein_g, fat_g) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (row['food_code'], row['food_name'], row['energy_kj'], row['energy_kcal'], row['carb_g'], row['protein_g'], row['fat_g'])
    )

# Commit and close connection
conn.commit()
cur.close()
conn.close()

print("CSV data imported successfully!")


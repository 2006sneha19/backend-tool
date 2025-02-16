import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="Nutrition_ai",   # Make sure this matches your database name
    user="postgres",
    password="Naruto@18#2",  # Replace with your PostgreSQL password
    host="localhost",
    port="5432"
)

# Create a cursor to execute queries
cur = conn.cursor()

# Fetch data from the table
query = "SELECT food_code, food_name, protein_g, fat_g FROM nutrition_csv_data;"
df = pd.read_sql(query, conn)

# Print the first 5 rows
print(df.head())

# Close connection
cur.close()
conn.close()

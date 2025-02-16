from fastapi import FastAPI
import pickle

# Load the model (replace with your model path)
with open("ai_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "AI tool backend is running!"}


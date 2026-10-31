from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("demand_model.pkl")
features = joblib.load("features.pkl")

class DemandInput(BaseModel):
    day_of_week: int
    attendance: int
    is_holiday: int
    prev_day_consumption: int
    menu_type: str  # "regular", "special" या "light"

@app.get("/")
def home():
    return {"message": "FoodSynk demand prediction API is running"}

@app.post("/predict")
def predict(data: DemandInput):
    row = {f: 0 for f in features}
    row["day_of_week"] = data.day_of_week
    row["attendance"] = data.attendance
    row["is_holiday"] = data.is_holiday
    row["prev_day_consumption"] = data.prev_day_consumption
    menu_col = "menu_type_" + data.menu_type
    if menu_col in row:
        row[menu_col] = 1
    df = pd.DataFrame([row])[features]
    result = model.predict(df)[0]
    return {"predicted_plates": round(float(result))}

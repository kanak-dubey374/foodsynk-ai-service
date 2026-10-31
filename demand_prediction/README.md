# Demand Prediction (FoodSynk)

An ML model and API that predict how many plates the canteen will need.

## Data
- File: `demand_data.csv` (400 rows, synthetic data)
- Columns: date, day_of_week, attendance, menu_type, is_holiday, prev_day_consumption, plates_consumed, plates_produced, surplus

## Model
- Algorithm: Random Forest Regressor (scikit-learn)
- Input: day_of_week, attendance, is_holiday, prev_day_consumption, menu_type
- Output: predicted plates (demand)
- Data split: 80% training, 20% testing

## Accuracy
- Mean Absolute Error: 8.7 plates
- R2 Score: 0.986

## Files
- `app.py`: FastAPI code
- `demand_model.pkl`: trained model
- `features.pkl`: list of input columns the model expects
- `requirements.txt`: required libraries
- `01_data_generation_ipynb.ipynb`: notebook (training code)

## How to Run the API
1. `pip install -r requirements.txt`
2. `python -m uvicorn app:app --reload`
3. Open in browser: http://127.0.0.1:8000/docs

## Example
Input:
{"day_of_week": 2, "attendance": 450, "is_holiday": 0, "prev_day_consumption": 400, "menu_type": "regular"}

Output:
{"predicted_plates": 382}

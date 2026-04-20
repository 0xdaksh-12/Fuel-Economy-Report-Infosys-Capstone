import joblib
import pandas as pd

new_vehicle = pd.DataFrame([{
  'Fuel': 'Gasoline',
  'Engine_Cylinders': 4,
  'Greenhouse_Gas_Score': 3,
  'Air_Pollution_Score': 4,
  'Combined_Mpg': 10,
  'Engine_Displacement_L': 3.7,
  'Drive': 4,
  
}])

model = joblib.load('./model/smartway_decision_tree_model.pkl')

# Predict
prediction = model.predict(new_vehicle)

label_map = {0: "No", 1: "Yes", 2: "Elite"}
print("Prediction:", label_map[prediction[0]])

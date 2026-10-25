
import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load("model.pkl")

# 2. Load held-out passenger records
df = pd.read_csv("unseen_passengers.csv")

# 3. Select one unseen passenger
passenger = df.iloc[[0]]

# Keep the passenger ID for display
passenger_id = passenger["PassengerId"].iloc[0]

# 4. Select only the model input features
features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

X = passenger[features]

# 5. Generate prediction
prediction = model.predict(X)[0]

# 6. Display the result
print("ONLINE INFERENCE")
print("-" * 30)
print("Passenger ID:", passenger_id)
print("Passenger Details:")
print(X.to_string(index=False))

print("\nPredicted Survival:", int(prediction))

if prediction == 1:
    print("Result: Predicted to have survived")
else:
    print("Result: Predicted not to have survived")

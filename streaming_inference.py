
import pandas as pd
import joblib
import time

# 1. Load the saved model
model = joblib.load("model.pkl")

# 2. Load unseen passenger records
df = pd.read_csv("unseen_passengers.csv")

# 3. Select 10 passengers
stream = df.head(10).copy()

# 4. Define model input features
features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

print("STREAMING INFERENCE")
print("=" * 40)
print("Simulating incoming passenger records...\n")

# 5. Process one passenger at a time
for index, passenger in stream.iterrows():

    passenger_id = passenger["PassengerId"]

    # Prepare one record as a DataFrame
    X = passenger[features].to_frame().T

    # Generate prediction
    prediction = int(model.predict(X)[0])

    # Display result
    print(f"Passenger ID: {passenger_id}")
    print(f"Predicted Survival: {prediction}")

    if prediction == 1:
        print("Result: Predicted to have survived")
    else:
        print("Result: Predicted not to have survived")

    print("-" * 30)

    # Wait one second before processing the next record,
    # except after the last record
    if index != stream.index[-1]:
        time.sleep(1)

print("\nSTREAMING INFERENCE COMPLETED!")

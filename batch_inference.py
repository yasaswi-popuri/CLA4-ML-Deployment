
import pandas as pd
import joblib

# 1. Load the saved trained pipeline
model = joblib.load("model.pkl")

# 2. Load unseen passenger records
df = pd.read_csv("unseen_passengers.csv")

# 3. Select 20 passengers
batch = df.head(20).copy()

# 4. Keep passenger IDs separately
passenger_ids = batch["PassengerId"]

# 5. Select the model input features
features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

X = batch[features]

# 6. Predict for all 20 passengers together
predictions = model.predict(X)

# 7. Create the results DataFrame
results = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Predicted_Survived": predictions.astype(int)
})

# 8. Save predictions to CSV
results.to_csv("batch_predictions.csv", index=False)

# 9. Display results
print("BATCH INFERENCE")
print("-" * 40)
print("Total passengers processed:", len(results))
print("\nPredictions:")
print(results.to_string(index=False))

print("\nSaved results to batch_predictions.csv")
print("\nBATCH INFERENCE COMPLETED SUCCESSFULLY!")

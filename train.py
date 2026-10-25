
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Titanic dataset
df = pd.read_csv("train.csv")

print("TITANIC ML MODEL TRAINING")
print("-" * 35)
print("Dataset shape:", df.shape)

# 2. Select input features and target
feature_columns = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

X = df[feature_columns]
y = df["Survived"]

# Keep passenger IDs only for identifying inference examples
passenger_ids = df["PassengerId"]

# 3. Split into training and testing portions
X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        y,
        passenger_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Preprocess numerical and categorical features
numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = ["Sex", "Embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Create the complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 6. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# 7. Evaluate using held-out testing data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Model Accuracy Percentage: {accuracy * 100:.2f}%")

# 8. Save the complete pipeline
joblib.dump(model, "model.pkl")
print("\nSaved trained pipeline as model.pkl")

# 9. Save 31 held-out records for later inference tasks
# These records were not used to train the model.
unseen = X_test.head(31).copy()
unseen.insert(0, "PassengerId", id_test.loc[unseen.index])

unseen.to_csv("unseen_passengers.csv", index=False)
print("Saved held-out passenger records as unseen_passengers.csv")

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")

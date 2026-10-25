
import requests

# FastAPI prediction endpoint
url = "http://127.0.0.1:8000/predict"

# Passenger details
passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

try:
    # Send request to FastAPI
    response = requests.post(url, json=passenger, timeout=10)

    print("FASTAPI PYTHON CLIENT")
    print("-" * 35)
    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    print("Passenger Details:", passenger)
    print("Prediction Response:")
    print(response.json())

    print("\nCLIENT TEST COMPLETED SUCCESSFULLY!")

except requests.exceptions.RequestException as error:
    print("API request failed:", error)
    print("Check that the FastAPI server is running.")

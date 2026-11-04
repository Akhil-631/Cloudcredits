from src.predict import predict_survival


# Sample passenger details
passenger = {
    "pclass": 1,
    "sex": "male",
    "age": 25,
    "sibsp": 0,
    "parch": 1,
    "fare": 32.0,
    "embarked": "S"
}


# Generate prediction
result = predict_survival(passenger)


# Display results
print("\nTitanic Survival Prediction")
print("-" * 35)

print("Prediction:", result["prediction_label"])
print("Survival Probability:",
      f'{result["survival_probability"]:.2%}')

print("Non-Survival Probability:",
      f'{result["non_survival_probability"]:.2%}')
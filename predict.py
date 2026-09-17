
import joblib
import pandas as pd


MODEL_PATH = "spondyloarthropathy_model.pkl"
MODEL_INFO_PATH = "spondyloarthropathy_model_info.pkl"


model = joblib.load(MODEL_PATH)
model_info = joblib.load(MODEL_INFO_PATH)

FEATURES = model_info["features"]


def validate_patient_data(patient_data):
    required_features = set(FEATURES)
    provided_features = set(patient_data.keys())

    missing_features = required_features - provided_features
    extra_features = provided_features - required_features

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {sorted(extra_features)}"
        )

    numerical_features = [
        "Age",
        "Morning_Stiffness_Min",
        "ESR"
    ]

    for feature in numerical_features:
        if not isinstance(patient_data[feature], (int, float)):
            raise TypeError(
                f"{feature} must be a number."
            )

    if patient_data["Age"] < 0:
        raise ValueError("Age cannot be negative.")

    if patient_data["Morning_Stiffness_Min"] < 0:
        raise ValueError(
            "Morning_Stiffness_Min cannot be negative."
        )

    if patient_data["ESR"] < 0:
        raise ValueError("ESR cannot be negative.")

    return pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )


def predict_diagnosis(patient_data):
    patient_df = validate_patient_data(patient_data)

    prediction = model.predict(patient_df)[0]
    probabilities = model.predict_proba(patient_df)[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    return prediction, probability_dict

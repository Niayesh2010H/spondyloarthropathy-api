
from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_diagnosis


app = FastAPI(
    title="Spondyloarthropathy Diagnosis API",
    description="Machine Learning API for spondyloarthropathy classification",
    version="1.0.0"
)


class PatientData(BaseModel):
    Gender: str
    Age: float
    Inflammatory_Back_Pain: str
    Peripheral_Arthritis: str
    Joint_Involvement_Pattern: str
    Enthesitis: str
    Dactylitis: str
    Psoriasis: str
    IBD: str
    Recent_Infection: str
    Uveitis: str
    HLA_B27: str
    Family_History: str
    Radiologic_View: str
    Morning_Stiffness_Min: float
    ESR: float


@app.get("/")
def home():
    return {
        "message": "Spondyloarthropathy API is running"
    }


@app.post("/predict")
def predict(patient_data: PatientData):

    patient_dict = patient_data.model_dump()

    prediction, probabilities = predict_diagnosis(patient_dict)

    return {
        "prediction": prediction,
        "probabilities": probabilities
    }

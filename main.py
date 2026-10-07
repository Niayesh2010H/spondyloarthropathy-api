from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Spondyloarthropathy AI Analysis API",
    description="AI analysis API for spondyloarthropathy research prototype",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "message": "Spondyloarthropathy AI API is running"
    }


@app.post("/predict")
def predict(patient_data: PatientData):

    patient = patient_data.model_dump()

    # Temporary research/demo analysis.
    # This does NOT use a real medical AI model.

    findings = []

    if patient["Inflammatory_Back_Pain"] == "Yes":
        findings.append("Inflammatory back pain is present.")

    if patient["Enthesitis"] == "Yes":
        findings.append("Enthesitis is present.")

    if patient["Uveitis"] == "Yes":
        findings.append("Uveitis is present.")

    if patient["HLA_B27"] == "Positive":
        findings.append("HLA-B27 is reported as positive.")

    if patient["Psoriasis"] == "Yes":
        findings.append("Psoriasis is present.")

    if patient["IBD"] == "Yes":
        findings.append("Inflammatory bowel disease is reported.")

    if patient["Recent_Infection"] == "Yes":
        findings.append("A recent infection is reported.")

    if patient["Dactylitis"] == "Yes":
        findings.append("Dactylitis is present.")

    if patient["Peripheral_Arthritis"] == "Yes":
        findings.append("Peripheral arthritis is present.")

    if not findings:
        findings.append("No selected positive findings were identified.")

    analysis = {
        "status": "demo",
        "predicted_classification": "Requires Further Evaluation",
        "relative_likelihood": "Not determined",
        "important_findings": findings,
        "clinical_reasoning": (
            "This is a technical research prototype. "
            "The provided information is summarized for demonstration "
            "and does not constitute a medical diagnosis."
        ),
        "medical_disclaimer": (
            "This prototype is not a medical device and must not be used "
            "to diagnose or treat patients. Clinical decisions require "
            "evaluation by a qualified healthcare professional."
        )
    }

    return {
        "analysis": analysis
    }

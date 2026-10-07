import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI(
    title="Spondyloarthropathy AI Analysis API",
    description="AI API for spondyloarthropathy analysis",
    version="2.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
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

    prompt = f"""
You are an AI assistant for a rheumatology research prototype.

Analyze the following patient information:

{patient}

The possible spondyloarthropathy classes are:
- AS = Ankylosing Spondylitis
- EnA = Enteropathic Arthritis
- PsA = Psoriatic Arthritis
- ReA = Reactive Arthritis
- Undifferentiated = Undifferentiated Spondyloarthropathy

Provide a careful, medically cautious research-oriented analysis.

Your response MUST contain:

1. Predicted Classification
2. Relative Likelihood
3. Important Findings
4. Clinical Reasoning
5. Medical Disclaimer

Do NOT invent numerical probabilities or claim diagnostic certainty.

This is a research prototype using user-provided information.
It is NOT a medical diagnosis and must NOT replace evaluation by a qualified healthcare professional.
"""

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You are a careful medical research assistant. "
            "Do not present your analysis as a definitive diagnosis. "
            "Do not invent missing clinical information."
        ),
        input=prompt
    )

    return {
        "analysis": response.output_text
    }

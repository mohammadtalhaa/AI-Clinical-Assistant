"""
AI Clinical Decision Assistant
================================
A production-quality AI medical application using:
- Streamlit (UI)
- RAG Pipeline (FAISS + sentence-transformers / OpenAI embeddings)
- OCR (PyMuPDF + pytesseract / easyocr)
- Rule-based Clinical Engine
- LLM Integration (OpenAI GPT-4 / fallback mock)
- Structured JSON output

Single-file implementation — all modules inlined for portability.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import re
import io
import json
import logging
import datetime
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("ClinicalAssistant")

# ─────────────────────────────────────────────
# OPTIONAL IMPORTS — graceful degradation
# ─────────────────────────────────────────────
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─────────────────────────────────────────────
# ██████╗  █████╗ ████████╗ █████╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# ██║  ██║███████║   ██║   ███████║
# ██║  ██║██╔══██║   ██║   ██╔══██║
# ██████╔╝██║  ██║   ██║   ██║  ██║
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
# MEDICAL KNOWLEDGE BASE (Mock / Sample)
# ─────────────────────────────────────────────
MEDICAL_KNOWLEDGE_BASE = [
    {
        "id": "pneumonia_01",
        "source": "WHO Guidelines – Pneumonia",
        "text": (
            "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
            "Key symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. "
            "Common causes include Streptococcus pneumoniae (bacterial), influenza virus, or fungi. "
            "Diagnosis involves chest X-ray, CBC showing elevated WBC, sputum culture. "
            "Treatment: antibiotics (amoxicillin / azithromycin for community-acquired), supportive care, oxygen if SpO2 < 94%. "
            "High-risk groups: elderly, immunocompromised, children under 5."
        ),
    },
    {
        "id": "pneumonia_02",
        "source": "IDSA/ATS Community-Acquired Pneumonia Guidelines",
        "text": (
            "CURB-65 score is used to assess pneumonia severity: Confusion, Urea > 7 mmol/L, "
            "Respiratory rate ≥ 30/min, Blood pressure (systolic < 90 or diastolic ≤ 60), Age ≥ 65. "
            "Score 0-1: low severity, outpatient treatment. Score 2: moderate, consider hospitalisation. "
            "Score 3+: severe, hospital admission required. Recommended tests: CBC, CMP, blood cultures, procalcitonin."
        ),
    },
    {
        "id": "diabetes_01",
        "source": "ADA Standards of Medical Care in Diabetes",
        "text": (
            "Type 2 Diabetes Mellitus is characterised by insulin resistance and relative insulin deficiency. "
            "Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L) on two occasions confirms diagnosis. "
            "HbA1c ≥ 6.5% is also diagnostic. Symptoms: polyuria, polydipsia, unexplained weight loss, blurred vision. "
            "First-line treatment: metformin + lifestyle modification. "
            "Monitor: HbA1c every 3 months, annual eye exam, foot exam, kidney function (eGFR, urine ACR)."
        ),
    },
    {
        "id": "diabetes_02",
        "source": "IDF Diabetes Atlas",
        "text": (
            "Diabetic emergencies include Diabetic Ketoacidosis (DKA) — blood glucose > 250 mg/dL, "
            "pH < 7.3, bicarbonate < 15, ketonemia/ketonuria. DKA is life-threatening and requires IV fluids, "
            "insulin infusion, electrolyte replacement. Hyperglycaemic Hyperosmolar State (HHS): glucose > 600 mg/dL, "
            "osmolarity > 320, no significant ketosis. Both require emergency hospitalisation."
        ),
    },
    {
        "id": "hypertension_01",
        "source": "JNC 8 / ESC/ESH Hypertension Guidelines",
        "text": (
            "Hypertension is defined as sustained blood pressure ≥ 130/80 mmHg (ACC/AHA 2017) or ≥ 140/90 mmHg (ESC/ESH). "
            "Stage 1: 130-139/80-89 mmHg. Stage 2: ≥ 140/90 mmHg. Hypertensive crisis: > 180/120 mmHg. "
            "Symptoms: often asymptomatic (silent killer). Severe: headache, vision changes, chest pain, shortness of breath. "
            "First-line agents: ACE inhibitors, ARBs, thiazide diuretics, CCBs. "
            "Lifestyle: reduce sodium < 2.3g/day, DASH diet, exercise 150 min/week, limit alcohol."
        ),
    },
    {
        "id": "hypertension_02",
        "source": "WHO Cardiovascular Disease Prevention Guidelines",
        "text": (
            "Hypertensive emergency: BP > 180/120 with end-organ damage (STEMI, stroke, AKI, hypertensive encephalopathy). "
            "Requires immediate IV antihypertensive therapy (labetalol, nicardipine, nitroprusside). "
            "Hypertensive urgency: BP > 180/120 without acute organ damage — oral agents, reduce BP over 24-48h. "
            "Recommended tests: ECG, renal function, urinalysis, fundoscopy, echocardiogram."
        ),
    },
    {
        "id": "flu_01",
        "source": "CDC Influenza Clinical Guidance",
        "text": (
            "Influenza (Flu) is caused by influenza A or B viruses. Symptoms appear 1-4 days after exposure. "
            "Key features: abrupt onset high fever (38-40°C), myalgia, headache, dry cough, sore throat, fatigue. "
            "Distinguishes from common cold by sudden onset and systemic symptoms. "
            "Diagnosis: rapid influenza test, RT-PCR. Treatment: oseltamivir (Tamiflu) within 48h of onset, "
            "supportive care (rest, fluids, antipyretics). Hospitalise if SpO2 < 94%, pneumonia, dehydration."
        ),
    },
    {
        "id": "flu_02",
        "source": "WHO Global Influenza Programme",
        "text": (
            "High-risk groups for influenza complications: age > 65, pregnancy, chronic lung/heart/kidney disease, "
            "diabetes, immunosuppression, obesity (BMI > 40). Annual vaccination recommended for all high-risk groups. "
            "Complications: viral pneumonia, secondary bacterial pneumonia, myocarditis, encephalitis, ARDS. "
            "Recommended tests: CBC (leukopenia common), CRP, chest X-ray if respiratory involvement."
        ),
    },
    {
        "id": "mi_01",
        "source": "ESC STEMI Guidelines 2023",
        "text": (
            "Acute Myocardial Infarction (Heart Attack) presents with chest pain/pressure radiating to left arm/jaw, "
            "diaphoresis (sweating), nausea, dyspnoea, and sense of doom. "
            "STEMI: ST elevation on ECG, requires immediate PCI (door-to-balloon < 90 min) or thrombolysis. "
            "NSTEMI: Elevated troponin without ST elevation, managed with anticoagulation + early cath. "
            "Key labs: Troponin I/T (rises in 3-6h, peaks 24h), CK-MB, BNP. ECG mandatory. "
            "Immediate management: aspirin 300mg, sublingual nitrates, oxygen if SpO2 < 94%, morphine for pain."
        ),
    },
    {
        "id": "mi_02",
        "source": "AHA/ACC Chest Pain Evaluation Guidelines",
        "text": (
            "Chest pain differential includes: ACS (STEMI/NSTEMI/UA), aortic dissection, pulmonary embolism, "
            "pneumothorax, pericarditis, oesophageal spasm. Red flags requiring emergency: "
            "tearing chest pain radiating to back (dissection), sudden pleuritic pain + dyspnoea (PE/pneumothorax), "
            "crushing chest pain + diaphoresis (ACS). HEART score used for risk stratification in chest pain."
        ),
    },
    {
        "id": "anemia_01",
        "source": "WHO Anaemia Guidelines",
        "text": (
            "Anaemia defined as Hb < 13 g/dL in adult males, < 12 g/dL in adult females, < 11 g/dL in pregnancy. "
            "Iron deficiency anaemia: most common, microcytic hypochromic (low MCV, low ferritin, high TIBC). "
            "Symptoms: fatigue, pallor, dyspnoea on exertion, palpitations, cold intolerance. "
            "B12/folate deficiency: macrocytic anaemia (high MCV), neurological symptoms in B12 deficiency. "
            "Treatment: oral iron (ferrous sulphate), dietary changes; B12 injections or high-dose oral B12."
        ),
    },
    {
        "id": "infection_01",
        "source": "Surviving Sepsis Campaign Guidelines",
        "text": (
            "Sepsis: life-threatening organ dysfunction caused by dysregulated response to infection. "
            "qSOFA criteria: respiratory rate ≥ 22/min, altered mentation, systolic BP ≤ 100 mmHg. "
            "Two qSOFA criteria = high risk, initiate sepsis bundle: blood cultures, lactate, IV fluids 30 mL/kg, "
            "broad-spectrum antibiotics within 1 hour, vasopressors if MAP < 65 mmHg. "
            "Septic shock: sepsis + vasopressor requirement to maintain MAP ≥ 65 + lactate > 2 mmol/L. ICU admission required."
        ),
    },
]

# ─────────────────────────────────────────────
# ██████╗  █████╗  ██████╗
# ██╔══██╗██╔══██╗██╔════╝
# ██████╔╝███████║██║  ███╗
# ██╔══██╗██╔══██║██║   ██║
# ██║  ██║██║  ██║╚██████╔╝
# ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
# RAG PIPELINE
# ─────────────────────────────────────────────

class SimpleEmbedder:
    """
    TF-IDF-style simple text embedder used when sentence-transformers
    and OpenAI are both unavailable. Pure Python, no dependencies.
    """
    def __init__(self):
        self.vocab: dict = {}
        self.idf: dict = {}
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def fit(self, corpus: list[str]):
        """Build vocab and IDF from corpus."""
        import math
        N = len(corpus)
        df: dict = {}
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self.vocab = {w: i for i, w in enumerate(sorted(df.keys()))}
        self.idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in df}
        self._fitted = True

    def embed(self, text: str) -> list[float]:
        """Return a normalised TF-IDF vector."""
        tokens = self._tokenize(text)
        vec = [0.0] * len(self.vocab)
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, count in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (count / len(tokens)) * self.idf.get(t, 1.0)
        # L2 normalise
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline.
    Uses FAISS (if available) or brute-force cosine similarity.
    Uses sentence-transformers (if available) or SimpleEmbedder.
    """

    def __init__(self, knowledge_base: list[dict], openai_api_key: Optional[str] = None):
        self.knowledge_base = knowledge_base
        self.openai_api_key = openai_api_key
        self.index = None
        self.embedder = None
        self.embeddings = None
        self._build_index()

    def _get_embedder(self):
        """Return best available embedder."""
        if SBERT_AVAILABLE:
            logger.info("Using sentence-transformers for embeddings.")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return ("sbert", model)
        elif self.openai_api_key and OPENAI_AVAILABLE:
            logger.info("Using OpenAI for embeddings.")
            return ("openai", OpenAI(api_key=self.openai_api_key))
        else:
            logger.info("Using SimpleEmbedder (TF-IDF fallback).")
            embedder = SimpleEmbedder()
            corpus = [doc["text"] for doc in self.knowledge_base]
            embedder.fit(corpus)
            return ("simple", embedder)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        kind, model = self.embedder
        if kind == "sbert":
            vecs = model.encode(texts, normalize_embeddings=True).tolist()
            return vecs
        elif kind == "openai":
            result = model.embeddings.create(input=texts, model="text-embedding-ada-002")
            return [r.embedding for r in result.data]
        else:  # simple
            return model.embed_batch(texts)

    def _build_index(self):
        """Chunk, embed, and index all documents."""
        self.embedder = self._get_embedder()
        texts = [doc["text"] for doc in self.knowledge_base]
        logger.info(f"Building RAG index over {len(texts)} documents...")
        raw_vecs = self._embed_texts(texts)

        if FAISS_AVAILABLE and NUMPY_AVAILABLE:
            dim = len(raw_vecs[0])
            self.index = faiss.IndexFlatIP(dim)  # Inner-product (cosine on normalised vecs)
            vecs_np = np.array(raw_vecs, dtype="float32")
            self.index.add(vecs_np)
            self.embeddings = vecs_np
            logger.info(f"FAISS index built: {dim}d, {self.index.ntotal} vectors.")
        else:
            # Store raw embeddings for brute-force cosine
            self.embeddings = raw_vecs
            logger.info("Brute-force cosine similarity (FAISS unavailable).")

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Retrieve top-k most relevant documents for a query."""
        query_vec = self._embed_texts([query])[0]

        if FAISS_AVAILABLE and NUMPY_AVAILABLE and self.index is not None:
            q_np = np.array([query_vec], dtype="float32")
            scores, indices = self.index.search(q_np, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    doc = dict(self.knowledge_base[idx])
                    doc["score"] = float(score)
                    results.append(doc)
        else:
            scored = [
                (self._cosine(query_vec, emb), i)
                for i, emb in enumerate(self.embeddings)
            ]
            scored.sort(reverse=True)
            results = []
            for score, idx in scored[:top_k]:
                doc = dict(self.knowledge_base[idx])
                doc["score"] = float(score)
                results.append(doc)

        logger.info(f"Retrieved {len(results)} docs for query snippet: '{query[:60]}...'")
        return results


# ─────────────────────────────────────────────
# OCR MODULE
# ─────────────────────────────────────────────

class OCRModule:
    """
    Extracts text from PDF or image files.
    Falls back through: PyMuPDF → Tesseract → EasyOCR → raw bytes attempt.
    Also parses common lab values from extracted text.
    """

    LAB_PATTERNS = {
        "Hemoglobin (g/dL)": r"h(?:e|a)moglobin[\s:]*([0-9]+\.?[0-9]*)",
        "Glucose (mg/dL)": r"glucose[\s:]*([0-9]+\.?[0-9]*)",
        "WBC (×10³/µL)": r"w(?:hite\s+blood\s+cell|bc)[\s:]*([0-9]+\.?[0-9]*)",
        "RBC (×10⁶/µL)": r"r(?:ed\s+blood\s+cell|bc)[\s:]*([0-9]+\.?[0-9]*)",
        "Platelets (×10³/µL)": r"platelet[\s:]*([0-9]+\.?[0-9]*)",
        "HbA1c (%)": r"hba1c[\s:%]*([0-9]+\.?[0-9]*)",
        "Systolic BP (mmHg)": r"(?:systolic|sbp|bp)[\s:]*([0-9]{2,3})[\s/]",
        "Diastolic BP (mmHg)": r"(?:diastolic|dbp)[/\s:]*([0-9]{2,3})",
        "Temperature (°C)": r"(?:temp(?:erature)?)[\s:]*([3-4][0-9]\.?[0-9]*)",
        "SpO2 (%)": r"(?:spo2|oxygen\s+sat(?:uration)?)[\s:]*([0-9]+\.?[0-9]*)",
        "Creatinine (mg/dL)": r"creatinine[\s:]*([0-9]+\.?[0-9]*)",
        "Sodium (mEq/L)": r"sodium[\s:]*([0-9]+\.?[0-9]*)",
        "Potassium (mEq/L)": r"potassium[\s:]*([0-9]+\.?[0-9]*)",
    }

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from uploaded file (PDF or image)."""
        ext = Path(filename).suffix.lower()
        text = ""

        if ext == ".pdf":
            text = self._extract_pdf(file_bytes)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            text = self._extract_image(file_bytes)
        else:
            # Try PDF first, then image
            text = self._extract_pdf(file_bytes)
            if not text.strip():
                text = self._extract_image(file_bytes)

        if not text.strip():
            text = "[Could not extract text from file. Please ensure it is a readable PDF or clear image.]"
        logger.info(f"OCR extracted {len(text)} chars from {filename}")
        return text

    def _extract_pdf(self, file_bytes: bytes) -> str:
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                pages = [page.get_text() for page in doc]
                return "\n".join(pages)
            except Exception as e:
                logger.warning(f"PyMuPDF error: {e}")

        # Fallback: try to decode as text
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _extract_image(self, file_bytes: bytes) -> str:
        if TESSERACT_AVAILABLE and PIL_AVAILABLE:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                return pytesseract.image_to_string(img)
            except Exception as e:
                logger.warning(f"Tesseract error: {e}")

        if EASYOCR_AVAILABLE:
            try:
                reader = easyocr.Reader(["en"], gpu=False)
                result = reader.readtext(file_bytes, detail=0)
                return " ".join(result)
            except Exception as e:
                logger.warning(f"EasyOCR error: {e}")

        return ""

    def parse_lab_values(self, text: str) -> dict:
        """Extract structured lab values from text using regex patterns."""
        text_lower = text.lower()
        parsed = {}
        for label, pattern in self.LAB_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    parsed[label] = float(match.group(1))
                except ValueError:
                    pass
        return parsed


# ─────────────────────────────────────────────
# CLINICAL RULES ENGINE
# ─────────────────────────────────────────────

class ClinicalRulesEngine:
    """
    Deterministic rule-based scoring system.
    Analyses symptoms + lab values and returns:
    - risk_score (0-100)
    - risk_level (Low/Medium/High/Emergency)
    - triggered_rules (list of matched conditions)
    - suspected_conditions (list)
    """

    RULES = [
        # Format: (name, risk_points, condition_lambda, description)
        # EMERGENCY rules
        {
            "name": "Chest pain + diaphoresis",
            "points": 40, "level": "EMERGENCY",
            "condition": lambda s, l: ("chest pain" in s or "chest pressure" in s) and ("sweat" in s or "diaphore" in s),
            "flag": "Possible Acute Coronary Syndrome — EMERGENCY",
            "conditions": ["Acute MI (STEMI/NSTEMI)", "Unstable Angina"],
        },
        {
            "name": "Very high blood pressure",
            "points": 35, "level": "EMERGENCY",
            "condition": lambda s, l: l.get("Systolic BP (mmHg)", 0) > 180 or l.get("Diastolic BP (mmHg)", 0) > 120,
            "flag": "Hypertensive Crisis (BP > 180/120) — EMERGENCY",
            "conditions": ["Hypertensive Emergency", "Hypertensive Urgency"],
        },
        {
            "name": "Very low SpO2",
            "points": 40, "level": "EMERGENCY",
            "condition": lambda s, l: l.get("SpO2 (%)", 100) < 90,
            "flag": "Critical Hypoxia (SpO2 < 90%) — EMERGENCY",
            "conditions": ["Respiratory Failure", "Severe Pneumonia", "ARDS", "Pulmonary Embolism"],
        },
        {
            "name": "Extreme glucose — DKA concern",
            "points": 35, "level": "EMERGENCY",
            "condition": lambda s, l: l.get("Glucose (mg/dL)", 0) > 400,
            "flag": "Extreme Hyperglycaemia — Possible DKA/HHS — EMERGENCY",
            "conditions": ["Diabetic Ketoacidosis (DKA)", "Hyperglycaemic Hyperosmolar State (HHS)"],
        },
        # HIGH rules
        {
            "name": "Sepsis triad",
            "points": 30, "level": "HIGH",
            "condition": lambda s, l: (
                ("fever" in s or l.get("Temperature (°C)", 37) > 38.5) and
                ("confusion" in s or "altered" in s) and
                ("rapid breath" in s or "tachypnea" in s or l.get("SpO2 (%)", 100) < 94)
            ),
            "flag": "Possible Sepsis — Urgent Evaluation",
            "conditions": ["Sepsis", "Severe Infection"],
        },
        {
            "name": "High fever + cough + dyspnoea",
            "points": 25, "level": "HIGH",
            "condition": lambda s, l: (
                ("fever" in s or l.get("Temperature (°C)", 37) > 38.0) and
                ("cough" in s) and
                ("breath" in s or "dyspn" in s or "shortness" in s)
            ),
            "flag": "Likely Respiratory Infection (Pneumonia/Flu)",
            "conditions": ["Pneumonia", "Influenza", "COVID-19"],
        },
        {
            "name": "Elevated WBC + fever",
            "points": 20, "level": "HIGH",
            "condition": lambda s, l: l.get("WBC (×10³/µL)", 8) > 12 and (
                "fever" in s or l.get("Temperature (°C)", 37) > 38.0
            ),
            "flag": "Leukocytosis with Fever — Infection Likely",
            "conditions": ["Bacterial Infection", "Pneumonia", "Sepsis"],
        },
        {
            "name": "Severe anaemia",
            "points": 25, "level": "HIGH",
            "condition": lambda s, l: l.get("Hemoglobin (g/dL)", 14) < 7,
            "flag": "Severe Anaemia (Hb < 7 g/dL)",
            "conditions": ["Severe Iron Deficiency Anaemia", "Haemolytic Anaemia", "GI Bleed"],
        },
        # MEDIUM rules
        {
            "name": "High glucose — diabetes",
            "points": 15, "level": "MEDIUM",
            "condition": lambda s, l: l.get("Glucose (mg/dL)", 90) > 126 or (
                "polyuria" in s and "polydipsia" in s
            ),
            "flag": "Elevated Glucose — Diabetes Mellitus Suspected",
            "conditions": ["Type 2 Diabetes Mellitus", "Impaired Glucose Tolerance"],
        },
        {
            "name": "Elevated BP",
            "points": 15, "level": "MEDIUM",
            "condition": lambda s, l: l.get("Systolic BP (mmHg)", 120) >= 140 or l.get("Diastolic BP (mmHg)", 80) >= 90,
            "flag": "Hypertension Stage 2",
            "conditions": ["Essential Hypertension"],
        },
        {
            "name": "High HbA1c",
            "points": 15, "level": "MEDIUM",
            "condition": lambda s, l: l.get("HbA1c (%)", 5.0) >= 6.5,
            "flag": "HbA1c ≥ 6.5% — Diagnostic of Diabetes",
            "conditions": ["Type 2 Diabetes Mellitus"],
        },
        {
            "name": "Moderate anaemia",
            "points": 10, "level": "MEDIUM",
            "condition": lambda s, l: 7 <= l.get("Hemoglobin (g/dL)", 14) < 10,
            "flag": "Moderate Anaemia (Hb 7-10 g/dL)",
            "conditions": ["Iron Deficiency Anaemia", "Anaemia of Chronic Disease"],
        },
        # LOW rules
        {
            "name": "Fever + cough",
            "points": 8, "level": "LOW",
            "condition": lambda s, l: "fever" in s and "cough" in s,
            "flag": "Mild Febrile Illness — Likely Viral",
            "conditions": ["Influenza", "Upper Respiratory Tract Infection", "Common Cold"],
        },
        {
            "name": "Mild glucose elevation",
            "points": 5, "level": "LOW",
            "condition": lambda s, l: 100 <= l.get("Glucose (mg/dL)", 90) < 126,
            "flag": "Pre-Diabetes Range Glucose",
            "conditions": ["Pre-Diabetes", "Impaired Fasting Glucose"],
        },
        {
            "name": "Fatigue + pallor",
            "points": 5, "level": "LOW",
            "condition": lambda s, l: ("fatigue" in s or "tired" in s) and ("pallor" in s or "pale" in s),
            "flag": "Possible Mild Anaemia",
            "conditions": ["Iron Deficiency Anaemia", "B12/Folate Deficiency"],
        },
    ]

    def evaluate(self, symptoms_text: str, lab_values: dict) -> dict:
        """
        Run all rules against symptoms and lab values.
        Returns structured assessment.
        """
        s = symptoms_text.lower()
        l = lab_values

        triggered = []
        total_score = 0
        all_conditions = []
        highest_level = "LOW"
        level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

        for rule in self.RULES:
            try:
                if rule["condition"](s, l):
                    triggered.append({
                        "name": rule["name"],
                        "flag": rule["flag"],
                        "level": rule["level"],
                        "points": rule["points"],
                    })
                    total_score += rule["points"]
                    all_conditions.extend(rule["conditions"])
                    if level_order[rule["level"]] > level_order[highest_level]:
                        highest_level = rule["level"]
            except Exception:
                pass

        # Clamp score
        total_score = min(total_score, 100)

        # Deduplicate conditions and weight them
        condition_counts: dict = {}
        for c in all_conditions:
            condition_counts[c] = condition_counts.get(c, 0) + 1
        sorted_conditions = sorted(condition_counts.items(), key=lambda x: -x[1])

        return {
            "risk_score": total_score,
            "risk_level": highest_level,
            "triggered_rules": triggered,
            "suspected_conditions": sorted_conditions,
        }


# ─────────────────────────────────────────────
# LLM ENGINE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI Clinical Decision Support System embedded in a hospital information system.
You receive structured patient data: symptoms, lab values, rule-based risk assessment, and retrieved medical knowledge.
Your job is to synthesise all inputs and output a structured clinical assessment.

IMPORTANT CONSTRAINTS:
1. Always include the disclaimer that this is NOT a substitute for professional medical advice.
2. Never recommend specific drug doses without noting "as prescribed by physician".
3. If risk level is EMERGENCY, explicitly state the patient needs IMMEDIATE medical attention.
4. Be evidence-based and cite the provided context documents.
5. Output ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON.

OUTPUT FORMAT (strict JSON):
{
  "diagnoses": [
    {"name": "Disease Name", "probability": 0.0_to_1.0, "reasoning": "brief reason"}
  ],
  "risk_level": "LOW|MEDIUM|HIGH|EMERGENCY",
  "recommended_tests": ["test1", "test2"],
  "treatment_suggestions": ["suggestion1", "suggestion2"],
  "explanation": "Detailed clinical reasoning paragraph",
  "sources": ["source1", "source2"],
  "emergency_action": "null OR specific emergency instruction"
}"""


class LLMEngine:
    """
    LLM integration layer.
    Uses OpenAI GPT-4 if API key provided, else falls back to rule-based mock output.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI LLM initialised.")
        else:
            logger.info("Using mock LLM (no OpenAI API key).")

    def generate(
        self,
        symptoms: str,
        lab_values: dict,
        rule_assessment: dict,
        retrieved_docs: list[dict],
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> dict:
        """Generate clinical assessment. Returns structured dict."""

        if self.client:
            return self._openai_generate(symptoms, lab_values, rule_assessment, retrieved_docs, age, gender)
        else:
            return self._mock_generate(symptoms, lab_values, rule_assessment, retrieved_docs, age, gender)

    def _build_user_prompt(self, symptoms, lab_values, rule_assessment, retrieved_docs, age, gender):
        demo = ""
        if age:
            demo += f"Age: {age}"
        if gender:
            demo += f", Gender: {gender}"

        lab_str = "\n".join(f"  - {k}: {v}" for k, v in lab_values.items()) if lab_values else "  None provided"

        rules_str = "\n".join(
            f"  [{r['level']}] {r['flag']}" for r in rule_assessment["triggered_rules"]
        ) if rule_assessment["triggered_rules"] else "  No critical rules triggered"

        conditions_str = ", ".join(c for c, _ in rule_assessment["suspected_conditions"][:5]) or "None identified by rules"

        docs_str = "\n\n".join(
            f"[SOURCE: {d['source']} | Relevance: {d.get('score', 0):.2f}]\n{d['text']}"
            for d in retrieved_docs
        )

        return f"""PATIENT PRESENTATION:
{demo if demo else 'Demographics not provided'}

SYMPTOMS:
{symptoms}

LAB VALUES:
{lab_str}

RULE-BASED RISK ASSESSMENT:
Risk Score: {rule_assessment['risk_score']}/100
Risk Level: {rule_assessment['risk_level']}
Triggered Rules:
{rules_str}
Rule-Suspected Conditions: {conditions_str}

RETRIEVED MEDICAL KNOWLEDGE:
{docs_str}

Based on all the above, generate a comprehensive clinical assessment in the required JSON format."""

    def _openai_generate(self, symptoms, lab_values, rule_assessment, retrieved_docs, age, gender):
        user_prompt = self._build_user_prompt(symptoms, lab_values, rule_assessment, retrieved_docs, age, gender)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._mock_generate(symptoms, lab_values, rule_assessment, retrieved_docs, age, gender)

    def _mock_generate(self, symptoms, lab_values, rule_assessment, retrieved_docs, age, gender):
        """
        Deterministic mock that produces realistic output when no LLM is available.
        Uses rule assessment + retrieved docs to construct a plausible response.
        """
        conditions = rule_assessment["suspected_conditions"]
        risk = rule_assessment["risk_level"]
        score = rule_assessment["risk_score"]

        # Build diagnoses from suspected conditions
        diagnoses = []
        base_prob = 0.75
        for i, (cond, count) in enumerate(conditions[:4]):
            prob = max(0.05, base_prob - i * 0.15)
            diagnoses.append({
                "name": cond,
                "probability": round(prob, 2),
                "reasoning": f"Supported by {count} clinical indicator(s) and symptom pattern.",
            })

        if not diagnoses:
            diagnoses = [{"name": "Non-specific illness", "probability": 0.5, "reasoning": "Insufficient data for specific diagnosis."}]

        # Build recommended tests from condition context
        tests = set()
        if any("pneumonia" in c[0].lower() or "respiratory" in c[0].lower() for c in conditions):
            tests.update(["Chest X-ray", "CBC with differential", "Procalcitonin", "Blood cultures"])
        if any("diabet" in c[0].lower() or "glucose" in c[0].lower() for c in conditions):
            tests.update(["Fasting Glucose", "HbA1c", "Lipid Panel", "Urine ACR", "eGFR"])
        if any("hypertens" in c[0].lower() or "mi" in c[0].lower() or "cardiac" in c[0].lower() for c in conditions):
            tests.update(["ECG", "Troponin I/T", "BNP", "Renal Function", "Echocardiogram"])
        if any("anaemia" in c[0].lower() for c in conditions):
            tests.update(["CBC", "Serum Ferritin", "TIBC", "B12", "Folate", "Peripheral Blood Smear"])
        if not tests:
            tests = {"Full Blood Count (CBC)", "Comprehensive Metabolic Panel", "Urinalysis"}

        # Treatments
        treatments = ["Consult appropriate specialist", "Patient education and lifestyle counselling"]
        if risk in ("HIGH", "EMERGENCY"):
            treatments.insert(0, "⚠️ Urgent medical evaluation required")
        if any("diabet" in c[0].lower() for c in conditions):
            treatments.append("Dietary modification: low glycaemic index diet, regular exercise")
            treatments.append("Medication review — metformin or insulin as prescribed by physician")
        if any("hypertens" in c[0].lower() for c in conditions):
            treatments.append("Blood pressure monitoring daily")
            treatments.append("Sodium restriction < 2.3g/day, DASH diet")
        if any("pneumonia" in c[0].lower() or "flu" in c[0].lower() for c in conditions):
            treatments.append("Rest, adequate hydration, antipyretics as needed")
            treatments.append("Antibiotic/antiviral therapy as prescribed (if confirmed)")

        # Sources from retrieved docs
        sources = list({d["source"] for d in retrieved_docs[:3]})
        if not sources:
            sources = ["Clinical reasoning based on standard medical guidelines"]

        # Emergency action
        emergency_action = None
        if risk == "EMERGENCY":
            emergency_action = "CALL EMERGENCY SERVICES IMMEDIATELY (911 / 999 / 122). Do not delay. Arrange immediate hospital transport."

        # Explanation
        rules_desc = "; ".join(r["flag"] for r in rule_assessment["triggered_rules"][:3]) or "routine assessment"
        explanation = (
            f"Clinical analysis based on symptom review, lab values, and evidence-based guidelines. "
            f"Rule-based engine flagged: {rules_desc}. "
            f"Risk score computed at {score}/100 ({risk} level). "
            f"Retrieved relevant guidelines: {', '.join(sources[:2])}. "
            f"The leading diagnostic consideration is {diagnoses[0]['name']} given the presented clinical picture. "
            f"{'Urgent intervention and further workup are strongly recommended.' if risk in ('HIGH','EMERGENCY') else 'Follow-up and monitoring are recommended.'} "
            f"This assessment is generated by an AI system and must be reviewed by a qualified clinician."
        )

        return {
            "diagnoses": diagnoses,
            "risk_level": risk,
            "recommended_tests": sorted(tests)[:6],
            "treatment_suggestions": treatments[:5],
            "explanation": explanation,
            "sources": sources,
            "emergency_action": emergency_action,
        }


# ─────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────

def init_session():
    """Initialise all session state variables."""
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "rules_engine" not in st.session_state:
        st.session_state.rules_engine = ClinicalRulesEngine()
    if "ocr" not in st.session_state:
        st.session_state.ocr = OCRModule()
    if "history" not in st.session_state:
        st.session_state.history = []  # List of past assessments
    if "result" not in st.session_state:
        st.session_state.result = None
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "lab_values" not in st.session_state:
        st.session_state.lab_values = {}


def get_rag(api_key: Optional[str]) -> RAGPipeline:
    """Lazy-load RAG pipeline (cached in session)."""
    if st.session_state.rag is None:
        with st.spinner("🔬 Building medical knowledge index..."):
            st.session_state.rag = RAGPipeline(MEDICAL_KNOWLEDGE_BASE, openai_api_key=api_key)
    return st.session_state.rag


def get_llm(api_key: Optional[str]) -> LLMEngine:
    """Lazy-load LLM engine (cached in session)."""
    if st.session_state.llm is None or (api_key and st.session_state.llm.api_key != api_key):
        st.session_state.llm = LLMEngine(api_key=api_key)
    return st.session_state.llm


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

RISK_CONFIG = {
    "LOW": {"color": "#22c55e", "bg": "#f0fdf4", "border": "#86efac", "icon": "🟢", "label": "LOW RISK"},
    "MEDIUM": {"color": "#f59e0b", "bg": "#fffbeb", "border": "#fcd34d", "icon": "🟡", "label": "MEDIUM RISK"},
    "HIGH": {"color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "icon": "🔴", "label": "HIGH RISK"},
    "EMERGENCY": {"color": "#7c3aed", "bg": "#faf5ff", "border": "#c4b5fd", "icon": "🚨", "label": "⚠️ EMERGENCY"},
}


def risk_badge(level: str) -> str:
    cfg = RISK_CONFIG.get(level, RISK_CONFIG["LOW"])
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:2px solid {cfg["border"]};'
        f'padding:6px 16px;border-radius:20px;font-weight:700;font-size:1rem;">'
        f'{cfg["icon"]} {cfg["label"]}</span>'
    )


def prob_bar(name: str, prob: float, reasoning: str = ""):
    """Render a styled probability bar for a diagnosis."""
    pct = int(prob * 100)
    color = "#ef4444" if pct > 65 else "#f59e0b" if pct > 35 else "#22c55e"
    st.markdown(f"""
<div style="margin-bottom:14px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-weight:600;color:#1e293b;">{name}</span>
    <span style="font-weight:700;color:{color};">{pct}%</span>
  </div>
  <div style="background:#e2e8f0;border-radius:8px;height:10px;">
    <div style="background:{color};width:{pct}%;height:10px;border-radius:8px;
                transition:width 0.5s ease;"></div>
  </div>
  {"<p style='font-size:0.78rem;color:#64748b;margin-top:3px;'>" + reasoning + "</p>" if reasoning else ""}
</div>""", unsafe_allow_html=True)


def section_card(title: str, icon: str, content_fn, accent="#0f172a"):
    """Render a styled section card."""
    st.markdown(f"""
<div style="border-left:4px solid {accent};padding-left:12px;margin-bottom:6px;">
  <h3 style="color:{accent};margin:0;font-size:1.05rem;">{icon} {title}</h3>
</div>""", unsafe_allow_html=True)
    content_fn()


# ─────────────────────────────────────────────
# MAIN STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    # ── Page config ──────────────────────────
    st.set_page_config(
        page_title="AI Clinical Decision Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()

    # ── Global CSS ───────────────────────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    color: #f8fafc;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 6px 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}
.header-badge {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    color: #60a5fa;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 8px;
    border: 1px solid rgba(59,130,246,0.3);
}

/* Cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* Disclaimer */
.disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #92400e;
    margin-bottom: 1.5rem;
}

/* Emergency banner */
.emergency-banner {
    background: linear-gradient(135deg, #7c3aed, #dc2626);
    color: white;
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.05rem;
    animation: pulse 1.5s infinite;
    margin-bottom: 1.5rem;
    text-align: center;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

/* Source tags */
.source-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 3px 4px 3px 0;
}

/* Test tag */
.test-tag {
    display: inline-block;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px 4px;
}

/* Stagger */
.stagger { animation: fadeUp 0.5s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}

/* Override Streamlit button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown("""
<div style="padding:1rem 0 1.5rem;">
  <div style="font-size:1.6rem;margin-bottom:4px;">🏥</div>
  <div style="font-size:1rem;font-weight:700;color:#f8fafc;">Clinical AI</div>
  <div style="font-size:0.72rem;color:#64748b;font-family:'IBM Plex Mono',monospace;">v2.1 · RAG Enabled</div>
</div>""", unsafe_allow_html=True)

        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            placeholder="sk-...",
            help="If provided, uses GPT-4o. Otherwise runs in offline mode with mock LLM.",
        )

        st.markdown("---")
        st.markdown("### 👤 Patient Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
        age = age if age > 0 else None
        gender = gender if gender != "Not specified" else None

        st.markdown("---")
        st.markdown("### 📊 System Status")

        def status(label, available):
            icon = "✅" if available else "⚠️"
            color = "#22c55e" if available else "#f59e0b"
            st.markdown(f'<span style="color:{color};font-size:0.82rem;">{icon} {label}</span>', unsafe_allow_html=True)

        status("FAISS Vector DB", FAISS_AVAILABLE)
        status("Sentence-Transformers", SBERT_AVAILABLE)
        status("PyMuPDF (PDF)", PYMUPDF_AVAILABLE)
        status("Tesseract OCR", TESSERACT_AVAILABLE)
        status("EasyOCR", EASYOCR_AVAILABLE)
        status("OpenAI SDK", OPENAI_AVAILABLE)
        status("NumPy", NUMPY_AVAILABLE)

        st.markdown("---")
        if st.session_state.history:
            st.markdown(f"### 🗂️ Session History ({len(st.session_state.history)} record(s))")
            for i, h in enumerate(reversed(st.session_state.history[-5:])):
                risk = h["result"].get("risk_level", "?")
                color = RISK_CONFIG.get(risk, RISK_CONFIG["LOW"])["color"]
                st.markdown(
                    f'<div style="font-size:0.78rem;padding:4px 0;border-bottom:1px solid #1e293b;">'
                    f'<span style="color:{color};font-weight:700;">[{risk}]</span> '
                    f'<span style="color:#94a3b8;">{h["timestamp"]}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Main Header ──────────────────────────
    st.markdown("""
<div class="main-header">
  <div>
    <span class="header-badge">RAG · PIPELINE</span>
    <span class="header-badge">RULE · ENGINE</span>
    <span class="header-badge">LLM · POWERED</span>
  </div>
  <h1>🏥 AI Clinical Decision Assistant</h1>
  <p>Evidence-based · Retrieval-Augmented · Hybrid AI Reasoning</p>
</div>
""", unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────
    st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for educational and decision-support purposes ONLY.
  It does <strong>NOT</strong> constitute medical advice, diagnosis, or treatment. Always consult a licensed
  healthcare professional. In a medical emergency, call your local emergency services immediately.
</div>
""", unsafe_allow_html=True)

    # ── Input Section ─────────────────────────
    col_input, col_upload = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown("### 🩺 Symptom Input")
        symptoms = st.text_area(
            "Describe the patient's symptoms in detail",
            placeholder=(
                "e.g. 'Patient presents with high fever (39.2°C) for 3 days, "
                "productive cough with yellowish sputum, shortness of breath, "
                "chest pain on inspiration, fatigue, and loss of appetite. "
                "No known drug allergies. Non-smoker. No recent travel.'"
            ),
            height=220,
            label_visibility="collapsed",
        )

        # Quick symptom chips
        st.markdown("**Quick-add common symptoms:**")
        qcols = st.columns(4)
        quick = ["Fever", "Cough", "Chest pain", "Dyspnoea", "Fatigue", "Sweating", "Headache", "Nausea"]
        selected_quick = []
        for i, sym in enumerate(quick):
            with qcols[i % 4]:
                if st.button(sym, key=f"quick_{i}", use_container_width=True):
                    selected_quick.append(sym.lower())

        if selected_quick:
            # Append to symptoms text
            appended = symptoms + " " + ", ".join(selected_quick)
            symptoms = appended

    with col_upload:
        st.markdown("### 📎 Medical Report Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF or image (optional)",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            help="Medical report, lab results, or clinical documents.",
            label_visibility="collapsed",
        )

        if uploaded_file:
            with st.spinner("🔍 Extracting text with OCR..."):
                file_bytes = uploaded_file.read()
                ocr = st.session_state.ocr
                text = ocr.extract_text(file_bytes, uploaded_file.name)
                labs = ocr.parse_lab_values(text)
                st.session_state.ocr_text = text
                st.session_state.lab_values = labs

            st.success(f"✅ Extracted {len(text)} characters")

            if labs:
                st.markdown("**📋 Parsed Lab Values:**")
                for k, v in labs.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:0.85rem;">'
                        f'<span style="color:#475569;">{k}</span>'
                        f'<span style="font-weight:600;color:#0f172a;">{v}</span></div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("📄 View extracted text"):
                st.text_area("", value=st.session_state.ocr_text, height=200, disabled=True, label_visibility="collapsed")

        # Manual lab value entry
        with st.expander("🔬 Enter Lab Values Manually"):
            col_a, col_b = st.columns(2)
            manual_labs = {}
            with col_a:
                hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
                if hb > 0:
                    manual_labs["Hemoglobin (g/dL)"] = hb
                gluc = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
                if gluc > 0:
                    manual_labs["Glucose (mg/dL)"] = gluc
                wbc = st.number_input("WBC (×10³/µL)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                if wbc > 0:
                    manual_labs["WBC (×10³/µL)"] = wbc
                hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
                if hba1c > 0:
                    manual_labs["HbA1c (%)"] = hba1c
            with col_b:
                sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=0, step=1)
                if sbp > 0:
                    manual_labs["Systolic BP (mmHg)"] = sbp
                dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=0, step=1)
                if dbp > 0:
                    manual_labs["Diastolic BP (mmHg)"] = dbp
                temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=30.0, step=0.1)
                if temp > 30:
                    manual_labs["Temperature (°C)"] = temp
                spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=0, step=1)
                if spo2 > 0:
                    manual_labs["SpO2 (%)"] = spo2

            # Merge manual labs (manual overrides OCR)
            st.session_state.lab_values.update(manual_labs)

    # ── Analyse Button ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_cols = st.columns([1, 2, 1])
    with btn_cols[1]:
        analyse_clicked = st.button("🔍 Analyse Patient — Run Clinical AI", use_container_width=True)

    # ── Analysis Pipeline ─────────────────────
    if analyse_clicked:
        if not symptoms.strip():
            st.warning("⚠️ Please enter patient symptoms before analysing.")
        else:
            # Combine symptom sources
            full_symptoms = symptoms
            if st.session_state.ocr_text:
                full_symptoms += "\n\nExtracted from report:\n" + st.session_state.ocr_text[:2000]

            lab_values = st.session_state.lab_values

            with st.spinner("🧠 Running hybrid AI analysis..."):
                # Step 1: Rule-based
                logger.info("Step 1: Rule-based clinical scoring")
                rules_engine = st.session_state.rules_engine
                rule_assessment = rules_engine.evaluate(full_symptoms, lab_values)

                # Step 2: RAG retrieval
                logger.info("Step 2: RAG retrieval")
                rag = get_rag(api_key if api_key else None)
                retrieved_docs = rag.retrieve(full_symptoms, top_k=4)

                # Step 3: LLM generation
                logger.info("Step 3: LLM generation")
                llm = get_llm(api_key if api_key else None)
                result = llm.generate(
                    symptoms=full_symptoms,
                    lab_values=lab_values,
                    rule_assessment=rule_assessment,
                    retrieved_docs=retrieved_docs,
                    age=age,
                    gender=gender,
                )

            st.session_state.result = result
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "symptoms_snippet": symptoms[:80],
                "result": result,
                "rule_assessment": rule_assessment,
            })
            logger.info(f"Analysis complete. Risk: {result.get('risk_level')}")

    # ── Results Display ───────────────────────
    if st.session_state.result:
        result = st.session_state.result
        rule_assessment = st.session_state.history[-1]["rule_assessment"] if st.session_state.history else {}

        st.markdown("---")
        st.markdown("## 📊 Clinical Assessment Results")

        # Emergency banner
        if result.get("emergency_action"):
            st.markdown(f"""
<div class="emergency-banner">
  🚨 EMERGENCY ALERT: {result["emergency_action"]}
</div>""", unsafe_allow_html=True)

        # Risk level + overview row
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        risk = result.get("risk_level", "LOW")
        cfg = RISK_CONFIG.get(risk, RISK_CONFIG["LOW"])

        with r_col1:
            st.markdown(f"""
<div class="metric-card" style="border-top:4px solid {cfg['color']};">
  <div style="font-size:0.75rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;">Risk Level</div>
  <div style="font-size:1.5rem;font-weight:700;color:{cfg['color']};margin-top:4px;">{cfg['icon']} {risk}</div>
</div>""", unsafe_allow_html=True)

        with r_col2:
            score = rule_assessment.get("risk_score", 0)
            st.markdown(f"""
<div class="metric-card">
  <div style="font-size:0.75rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;">Rule Score</div>
  <div style="font-size:1.5rem;font-weight:700;color:#0f172a;margin-top:4px;">{score} <span style="font-size:0.9rem;color:#94a3b8;">/ 100</span></div>
</div>""", unsafe_allow_html=True)

        with r_col3:
            n_diag = len(result.get("diagnoses", []))
            st.markdown(f"""
<div class="metric-card">
  <div style="font-size:0.75rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;">Diagnoses Found</div>
  <div style="font-size:1.5rem;font-weight:700;color:#0f172a;margin-top:4px;">{n_diag}</div>
</div>""", unsafe_allow_html=True)

        with r_col4:
            n_tests = len(result.get("recommended_tests", []))
            st.markdown(f"""
<div class="metric-card">
  <div style="font-size:0.75rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;">Tests Suggested</div>
  <div style="font-size:1.5rem;font-weight:700;color:#0f172a;margin-top:4px;">{n_tests}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main results columns
        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            # Diagnoses
            st.markdown("""
<div style="border-left:4px solid #2563eb;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#1e40af;margin:0;font-size:1.05rem;">🔬 Differential Diagnoses</h3>
</div>""", unsafe_allow_html=True)

            for diag in result.get("diagnoses", []):
                prob_bar(diag["name"], diag["probability"], diag.get("reasoning", ""))

            st.markdown("<br>", unsafe_allow_html=True)

            # Explanation
            st.markdown("""
<div style="border-left:4px solid #0891b2;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#0e7490;margin:0;font-size:1.05rem;">🧠 Clinical Reasoning</h3>
</div>""", unsafe_allow_html=True)
            st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.2rem;
font-size:0.88rem;line-height:1.7;color:#334155;">
{result.get("explanation", "No explanation available.")}
</div>""", unsafe_allow_html=True)

            # Rule triggers
            if rule_assessment.get("triggered_rules"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
<div style="border-left:4px solid #dc2626;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#b91c1c;margin:0;font-size:1.05rem;">⚡ Triggered Clinical Rules</h3>
</div>""", unsafe_allow_html=True)
                for rule in rule_assessment["triggered_rules"]:
                    rc = RISK_CONFIG.get(rule["level"], RISK_CONFIG["LOW"])
                    st.markdown(f"""
<div style="background:{rc['bg']};border:1px solid {rc['border']};border-radius:8px;
padding:8px 14px;margin-bottom:6px;font-size:0.85rem;">
  <span style="color:{rc['color']};font-weight:700;">[{rule['level']}]</span>
  <span style="color:#374151;margin-left:8px;">{rule['flag']}</span>
  <span style="float:right;color:#9ca3af;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;">+{rule['points']} pts</span>
</div>""", unsafe_allow_html=True)

        with right_col:
            # Recommended Tests
            st.markdown("""
<div style="border-left:4px solid #059669;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#047857;margin:0;font-size:1.05rem;">🔬 Recommended Tests</h3>
</div>""", unsafe_allow_html=True)
            tests_html = "".join(
                f'<span class="test-tag">🧪 {t}</span>' for t in result.get("recommended_tests", [])
            )
            st.markdown(f'<div style="margin-bottom:1.5rem;">{tests_html}</div>', unsafe_allow_html=True)

            # Treatment Suggestions
            st.markdown("""
<div style="border-left:4px solid #7c3aed;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#6d28d9;margin:0;font-size:1.05rem;">💊 Treatment Guidance</h3>
</div>""", unsafe_allow_html=True)
            for suggestion in result.get("treatment_suggestions", []):
                is_warning = suggestion.startswith("⚠️")
                bg = "#fef2f2" if is_warning else "#faf5ff"
                border = "#fca5a5" if is_warning else "#ddd6fe"
                st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
padding:8px 14px;margin-bottom:6px;font-size:0.85rem;color:#374151;">
  {"" if is_warning else "→ "}{suggestion}
</div>""", unsafe_allow_html=True)

            # Sources
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
<div style="border-left:4px solid #0f172a;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#0f172a;margin:0;font-size:1.05rem;">📚 Evidence Sources</h3>
</div>""", unsafe_allow_html=True)
            sources_html = "".join(
                f'<span class="source-tag">📄 {s}</span>' for s in result.get("sources", [])
            )
            st.markdown(f'<div>{sources_html}</div>', unsafe_allow_html=True)

            # Lab values summary if available
            if st.session_state.lab_values:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
<div style="border-left:4px solid #f59e0b;padding-left:12px;margin-bottom:12px;">
  <h3 style="color:#b45309;margin:0;font-size:1.05rem;">🩸 Lab Values Used</h3>
</div>""", unsafe_allow_html=True)
                for k, v in st.session_state.lab_values.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
                        f'border-bottom:1px solid #f1f5f9;font-size:0.82rem;">'
                        f'<span style="color:#6b7280;">{k}</span>'
                        f'<span style="font-weight:600;color:#0f172a;">{v}</span></div>',
                        unsafe_allow_html=True,
                    )

        # Raw JSON expander
        with st.expander("🔧 View Raw JSON Output"):
            st.json(result)

        # Final disclaimer
        st.markdown("""
<div class="disclaimer" style="margin-top:2rem;">
  <strong>⚠️ Important:</strong> This AI-generated assessment is for clinical decision <em>support</em> only.
  It must be reviewed and validated by a licensed medical professional before any clinical action is taken.
  Do not use this output as a substitute for professional medical judgment. If you or a patient are
  experiencing a medical emergency, call emergency services immediately.
</div>
""", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;color:#94a3b8;">
  <div style="font-size:3rem;margin-bottom:1rem;">🏥</div>
  <div style="font-size:1.1rem;font-weight:500;color:#64748b;">Enter symptoms and click <strong>Analyse</strong> to begin</div>
  <div style="font-size:0.85rem;margin-top:0.5rem;font-family:'IBM Plex Mono',monospace;">
    RAG · Rule Engine · LLM · OCR — all ready
  </div>
    <div style="font-size:1.1rem;font-weight:500;color:#64748b;">Made by <strong>Mohammd</strong> Talha</div>

</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
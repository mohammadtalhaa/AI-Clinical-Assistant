# 🏥 AI Clinical Decision Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)  
**RAG‑powered · Rule Engine · OCR · LLM Integration**  
*For educational and demonstration purposes only – NOT for actual medical use.*

---

## 📌 Overview

This is a **production‑style** clinical decision support system built with Streamlit. It combines:

- 🔍 **OCR** – Extracts text from PDFs and images (PyMuPDF + Tesseract)
- 📚 **RAG (Retrieval‑Augmented Generation)** – FAISS + sentence‑transformers over a medical knowledge base
- ⚙️ **Rule‑based Clinical Engine** – Scores symptoms and labs (LOW → EMERGENCY)
- 🧠 **LLM Integration** – OpenAI GPT‑4o (optional) with a realistic mock fallback

The system outputs a structured JSON assessment: differential diagnoses, risk level, recommended tests, treatment suggestions, and evidence sources.

---

## ⚠️ Important Disclaimer

> **This tool is NOT a substitute for professional medical advice, diagnosis, or treatment.**  
> It is intended **strictly for educational, research, and demonstration purposes**.  
> In a real medical emergency, call your local emergency services immediately.  
> The authors assume no liability for any use or misuse of this software.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **OCR upload** | PDF, PNG, JPEG – extracts text and parses common lab values (Hb, glucose, WBC, BP, SpO₂, etc.) |
| 🧬 **Rule engine** | 15+ clinical rules (ACS, sepsis, pneumonia, DKA, hypertensive crisis) – calculates risk score & level |
| 🔎 **RAG retrieval** | FAISS vector index over WHO/IDSA/ADA‑style guidelines – finds relevant medical context |
| 🤖 **LLM synthesis** | OpenAI GPT‑4o or offline mock LLM produces final structured assessment |
| 📊 **Risk badges** | Color‑coded output (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH / 🚨 EMERGENCY) |
| 🧪 **Lab entry** | Manual entry or auto‑parsed from reports |
| 🗂️ **Session history** | Keeps track of recent analyses |

---

## 🚀 Live Demo (Streamlit Cloud)

Once deployed, your app will be available at:  
`https://your-username-streamlit-app-name.streamlit.app`

(Replace with your actual URL)

---

## 💻 Local Installation

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) OpenAI API key – [get one here](https://platform.openai.com/api-keys)

### Step 1: Clone the repository

```bash
git clone https://github.com/mohammadtalhaa/ai-clinical-assistant.git
cd ai-clinical-assistant

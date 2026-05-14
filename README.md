# CivicVoice Bhutan 🇧🇹

A voice-enabled AI assistant for Bhutan's citizen-facing government services. Ask a question about a government service — by voice or text — and get a clear, accurate answer drawn from official Bhutan government documents.

**[Live Demo](https://huggingface.co/spaces/nibhatt10/civicvoice-bhutan)** · **[GitHub](https://github.com/nidhi-bhatt/civicvoice_bhutan)**

---

## What it does

Citizens can ask questions like:
- *"How do I register my newborn's birth?"*
- *"What documents do I need for a passport?"*
- *"How does the eSakor land portal work?"*

The system understands the question, figures out which government department handles it, searches real policy documents for the answer, and speaks it back.

---

## How it works

Voice/Text Input
↓
Whisper ASR — converts speech to text
↓
Llama 3.3 (Groq) — classifies which G2C service category it belongs to
↓
RAG (LangChain + FAISS) — searches real Bhutan government documents
↓
LLM — generates a clear, citizen-friendly answer
↓
gTTS — speaks the answer back in English

---

## Knowledge Base

All answers are grounded in real Bhutan government sources:

| Source | Content |
|--------|---------|
| Bhutan Digital Strategy 2024 | Digital governance and e-services |
| Bhutan AI Readiness Report 2024 | AI policy and readiness |
| Bhutan ICT Policy | ICT roadmap and strategy |
| DCRC (dcrc.moha.gov.bt) | Birth and death registration |
| MFA (mfa.gov.bt) | Passport and travel services |
| NLCS (esakor.nlcs.gov.bt) | Land and property (eSakor) |

---

## Supported Service Categories

- Birth & Death Registration
- Land & Property (eSakor)
- Passport & Travel
- Digital Services & NDI App
- General AI & Digital Policy

---

## Tech Stack

`Whisper` · `LangChain` · `FAISS` · `Llama 3.3 via Groq` · `gTTS` · `Gradio` · `HuggingFace Spaces`

---

## Note on Dzongkha

Dzongkha (Bhutan's national language) is absent from Google Translate, Common Voice, and FLEURS — making it one of the most under-resourced languages in NLP. This project operates in English while acknowledging that gap. A Dzongkha-capable pipeline would require a dedicated NMT model (e.g. from [nlp.cst.edu.bt](http://nlp.cst.edu.bt)) and labelled ASR data that does not yet publicly exist.

---

## Built for

This project was built as a portfolio contribution aligned with the [Omdena CivicAI Bhutan](https://omdena.com) project — exploring the feasibility of voice-enabled, multilingual AI for Bhutan's public services.

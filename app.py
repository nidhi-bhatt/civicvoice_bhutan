# ============================================================
# CivicVoice Bhutan — Step 4: Gradio UI
# ============================================================

import os
import gradio as gr
from gtts import gTTS
from faster_whisper import WhisperModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_PATH = "data/vectorstore"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

G2C_CATEGORIES = [
    "Birth & Death Registration",
    "Land & Property (eSakor)",
    "Passport & Travel",
    "Digital Services & NDI App",
    "General AI & Digital Policy",
]

SUGGESTED_QUESTIONS = [
    "How do I register my newborn's birth?",
    "What documents do I need for birth registration?",
    "How do I apply for a Bhutanese passport?",
    "Can I apply for a passport online?",
    "What is the Bhutan NDI app?",
    "How does the eSakor land portal work?",
]

# ============================================================
# LOAD MODELS AT STARTUP
# ============================================================

print("Loading models...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
client = Groq(api_key=GROQ_API_KEY)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print("✅ All models loaded")

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def classify_intent(question):
    prompt = f"""You are a Bhutan government service assistant.

A citizen asked: "{question}"

Which ONE of these G2C service categories does this belong to?
{chr(10).join(f'- {cat}' for cat in G2C_CATEGORIES)}

Reply with ONLY the category name, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def get_answer(question, intent):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(f"{intent}: {question}")
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful Bhutan government service assistant.

A citizen asked: "{question}"
Service category: {intent}

Use the following information from official Bhutan government documents to answer:
{context}

Give a clear, simple, step-by-step answer a citizen can understand.
Keep it under 150 words. If the information is not in the context, say so honestly."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def speak_english(text):
    os.makedirs("data/outputs", exist_ok=True)
    output_path = "data/outputs/answer_english.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)
    return output_path


def transcribe_audio(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([seg.text for seg in segments])
    return transcript.strip()

# ============================================================
# MAIN GRADIO FUNCTION
# ============================================================

def process(audio, text_question):
    # Get question from audio or text
    if audio is not None:
        question = transcribe_audio(audio)
    elif text_question and text_question.strip():
        question = text_question.strip()
    else:
        return "Please speak or type a question.", "", None

    # Run pipeline
    intent = classify_intent(question)
    answer = get_answer(question, intent)
    audio_out = speak_english(answer)

    return question, f"📂 {intent}", answer, audio_out

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="CivicVoice Bhutan") as demo:

    gr.Markdown("""
    # 🇧🇹 CivicVoice Bhutan
    ### AI-powered voice assistant for Bhutan citizen services
    Ask a question about government services — by voice or text.
    """)

    # Suggested questions
    gr.Markdown("### 💡 Suggested Questions")
    with gr.Row():
        btn1 = gr.Button(SUGGESTED_QUESTIONS[0], size="sm")
        btn2 = gr.Button(SUGGESTED_QUESTIONS[1], size="sm")
        btn3 = gr.Button(SUGGESTED_QUESTIONS[2], size="sm")
    with gr.Row():
        btn4 = gr.Button(SUGGESTED_QUESTIONS[3], size="sm")
        btn5 = gr.Button(SUGGESTED_QUESTIONS[4], size="sm")
        btn6 = gr.Button(SUGGESTED_QUESTIONS[5], size="sm")

    gr.Markdown("### 🎤 Your Question")
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your question"
        )
        text_input = gr.Textbox(
            label="Or type your question",
            placeholder="e.g. How do I register my newborn's birth?"
        )

    submit_btn = gr.Button("Get Answer 🔍", variant="primary")

    gr.Markdown("### 📋 Results")
    transcribed = gr.Textbox(label="Your Question (transcribed)")
    intent_out = gr.Textbox(label="Service Category Detected")
    answer_out = gr.Textbox(label="Answer", lines=6)
    audio_out = gr.Audio(label="Listen to Answer")

    # Button click handlers
    submit_btn.click(
        fn=process,
        inputs=[audio_input, text_input],
        outputs=[transcribed, intent_out, answer_out, audio_out]
    )

    # Suggested question buttons fill text box
    btn1.click(fn=lambda: SUGGESTED_QUESTIONS[0], outputs=text_input)
    btn2.click(fn=lambda: SUGGESTED_QUESTIONS[1], outputs=text_input)
    btn3.click(fn=lambda: SUGGESTED_QUESTIONS[2], outputs=text_input)
    btn4.click(fn=lambda: SUGGESTED_QUESTIONS[3], outputs=text_input)
    btn5.click(fn=lambda: SUGGESTED_QUESTIONS[4], outputs=text_input)
    btn6.click(fn=lambda: SUGGESTED_QUESTIONS[5], outputs=text_input)

    gr.Markdown("""
    ---
    Built with real Bhutan government data | 
    Sources: DCRC, MFA, NLCS, GovTech Bhutan
    """)

demo.launch()
# ============================================================
# CivicVoice Bhutan — Step 1: RAG Pipeline Setup
# ============================================================

import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATHS = [
    "data/pdfs/digital_strategy.pdf",
    "data/pdfs/ai_readiness_2024.pdf",
    "data/pdfs/bhutan_ict_policy.pdf",
]

WEB_URLS = [
    "https://dcrc.moha.gov.bt/index.php/birth-registration1/",
    "https://esakor.nlcs.gov.bt/faq_eSakor",
    "https://www.mfa.gov.bt/passport-application/"
]

VECTORSTORE_PATH = "data/vectorstore"

# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    docs = []

    print("Loading PDFs...")
    for path in PDF_PATHS:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            print(f"  ✅ Loaded: {path} ({len(pdf_docs)} pages)")
        else:
            print(f"  ⚠️  PDF not found: {path} — skipping")

    print("\nLoading web pages...")
    for url in WEB_URLS:
        try:
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            docs.extend(web_docs)
            print(f"  ✅ Loaded: {url}")
        except Exception as e:
            print(f"  ⚠️  Could not load {url}: {e}")

    print(f"\nTotal documents loaded: {len(docs)}")
    return docs

# ============================================================
# SPLIT INTO CHUNKS
# ============================================================

def split_documents(docs):
    print("\nSplitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vectorstore(chunks):
    print("\nCreating vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ Vector store saved to: {VECTORSTORE_PATH}")
    return vectorstore

# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded existing vector store from: {VECTORSTORE_PATH}")
    return vectorstore

# ============================================================
# TEST RETRIEVAL
# ============================================================

def test_retrieval(vectorstore):
    print("\n" + "="*50)
    print("TESTING RETRIEVAL")
    print("="*50)

    test_questions = [
        "How do I register my newborn's birth?",
        "What documents do I need for a passport?",
        "How does the eSakor land portal work?",
        "What is the Bhutan NDI app?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = retriever.invoke(question)
        if results:
            print(f"Top result preview:")
            print(f"  {results[0].page_content[:300]}...")
        print("-" * 40)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs("data/pdfs", exist_ok=True)

    if os.path.exists(VECTORSTORE_PATH):
        print("Vector store already exists — loading it...")
        vectorstore = load_vectorstore()
    else:
        docs = load_documents()
        if not docs:
            print("\n❌ No documents loaded. Check your PDF paths and internet connection.")
        else:
            chunks = split_documents(docs)
            vectorstore = create_vectorstore(chunks)

    test_retrieval(vectorstore)
    print("\nStep 1 complete! RAG pipeline is ready.")
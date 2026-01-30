import os
from dotenv import load_dotenv
import gradio as gr
import csv
import uuid
from datetime import datetime
import re
import pandas as pd

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# =========================================================
# Environment
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# =========================================================
# UI CSS (Mobile + Dark Mode)
# =========================================================
CUSTOM_CSS = """
body.dark-mode {
    background-color: #0f1117 !important;
    color: #e5e7eb !important;
}

body.dark-mode textarea,
body.dark-mode input,
body.dark-mode .gr-box,
body.dark-mode .gr-panel {
    background-color: #1c1f26 !important;
    color: #e5e7eb !important;
    border-color: #2a2f3a !important;
}

body.dark-mode button {
    background-color: #2563eb !important;
    color: white !important;
}

body.dark-mode .gr-dataframe {
    background-color: #111827 !important;
}

/* Mobile spacing */
@media (max-width: 768px) {
    .gradio-container { padding: 6px !important; }
    .gr-chatbot { font-size: 14px; }
    textarea { font-size: 14px !important; }
    button { min-height: 44px; }
    .gr-dataframe { font-size: 12px; overflow-x: auto; }
}
"""


# =========================================================
# Ticket System
# =========================================================
TICKET_FILE = "tickets/tickets.csv"

TICKET_COLUMNS = [
    "ticket_id",
    "timestamp",
    "message",
    "category",
    "priority",
    "status",
    "device",
    "ai_suggestion",
    "confidence",
    "similar_ticket"
]


def init_ticket_file():
    os.makedirs("tickets", exist_ok=True)
    if not os.path.exists(TICKET_FILE):
        with open(TICKET_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(TICKET_COLUMNS)


init_ticket_file()


def create_ticket(
    message,
    category,
    priority,
    device,
    ai_suggestion,
    confidence,
    similar_ticket
):
    ticket_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        ticket_id,
        timestamp,
        message,
        category,
        priority,
        "open",
        device,
        ai_suggestion,
        confidence,
        similar_ticket
    ]

    with open(TICKET_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return ticket_id


# =========================================================
# Fast Classification (Local)
# =========================================================
def fast_classify_ticket(text):
    text = text.lower()

    if any(word in text for word in ["crash", "error", "bug", "freeze"]):
        return "bug", "high"

    if any(word in text for word in ["lag", "slow", "fps", "stutter"]):
        return "performance", "medium"

    if any(word in text for word in ["how", "unlock", "play", "defeat"]):
        return "gameplay", "low"

    if any(word in text for word in ["login", "account", "payment", "purchase"]):
        return "account", "high"

    return "general", "medium"


# =========================================================
# Device Info Extraction
# =========================================================
def extract_device_info(text):
    patterns = [
        r"windows\s?\d+",
        r"rtx\s?\d+",
        r"gtx\s?\d+",
        r"\d+\s?gb\s?ram",
        r"intel\s?i\d",
        r"ryzen\s?\d",
        r"laptop",
        r"pc"
    ]

    matches = []
    text = text.lower()

    for p in patterns:
        found = re.findall(p, text)
        matches.extend(found)

    return ", ".join(set(matches)) if matches else "not provided"


# =========================================================
# Similar Ticket Engine (FAISS)
# =========================================================
ticket_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ticket_index = FAISS.from_texts(["placeholder"], ticket_embeddings)


def find_similar_ticket(text):
    try:
        results = ticket_index.similarity_search(text, k=1)
        if results and results[0].page_content != "placeholder":
            return results[0].page_content[:300]
    except:
        pass
    return "none"


def update_ticket_memory(text):
    ticket_index.add_texts([text])


# =========================================================
# LLM Setup (Groq)
# =========================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=GROQ_API_KEY
)


# =========================================================
# AI Auto Resolution Generator
# =========================================================
def generate_ai_solution(issue, similar_issue):
    prompt = f"""
You are a game technical support expert.

Issue:
{issue}

Similar Past Issue:
{similar_issue}

Provide:
1. A short possible fix (2–3 lines).
2. A confidence score between 0.30 and 0.95.
Format:
FIX: ...
CONFIDENCE: 0.xx
"""

    try:
        result = llm.invoke(prompt).content
        fix_match = re.search(r"FIX:(.*)", result)
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", result)

        fix = fix_match.group(1).strip() if fix_match else "Restart the game and verify files."
        confidence = conf_match.group(1).strip() if conf_match else "0.55"

        return fix, confidence
    except:
        return "Restart the game and update GPU drivers.", "0.50"


# =========================================================
# Ticket Dashboard Utilities
# =========================================================
def load_tickets():
    if not os.path.exists(TICKET_FILE):
        return pd.DataFrame(columns=TICKET_COLUMNS)

    df = pd.read_csv(TICKET_FILE)

    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].astype(str)

    return df


def filter_tickets(category, priority, status):
    df = load_tickets()

    if category != "All":
        df = df[df["category"] == category]

    if priority != "All":
        df = df[df["priority"] == priority]

    if status != "All":
        df = df[df["status"] == status]

    return df

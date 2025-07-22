from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
import spacy
from spacy.matcher import PhraseMatcher
import re
import json

from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
from backend_api import fetch_account_details

# Initialize FastAPI & Azure OpenAI client
app = FastAPI(title="Azure OpenAI Agent")
client = OpenAIClient(AZURE_OPENAI_ENDPOINT, AzureKeyCredential(AZURE_OPENAI_API_KEY))

# Load spaCy model and setup PhraseMatchers
nlp = spacy.load("en_core_web_sm")
intent_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
intent_patterns = [nlp(text) for text in ["fetch account details", "get account details", "account info"]]
intent_matcher.add("FETCH_ACCOUNT_DETAILS", intent_patterns)

ip_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
ip_patterns = [nlp(text) for text in ["ip", "ips", "whitelisted ip", "whitelist ip"]]
ip_matcher.add("IP_REQUEST", ip_patterns)

# In-memory sessions
sessions = {}

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

class AccountInfo(BaseModel):
    account_name: str = Field(..., regex=r"^[a-zA-Z0-9]+$")
    environment: str = Field(..., regex=r"^(prod|non prod|cgm prod)$")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sess = sessions.setdefault(req.session_id, {"state": "init"})
    text = req.message.strip()
    doc = nlp(text)

    # Initial Greeting
    if sess["state"] == "init":
        sess["state"] = "greeted"
        return {"reply": "Hi!! How may I help you?"}

    # Hybrid Intent: spaCy + OpenAI
    matches = intent_matcher(doc)
    if matches:
        intent = "FetchAccountDetails"
    else:
        # fallback refine via OpenAI zero-shot
        prompt = (
            "Classify this message into one of: FetchAccountDetails, General. "
            f"Message: '{text}'\nReply with intent name only."
        )
        resp = client.get_chat_completions(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        intent = resp.choices[0].message.content.strip()

    # FetchAccountDetails Flow
    if intent == "FetchAccountDetails" and sess["state"] == "greeted":
        sess["state"] = "awaiting_details"
        return {"reply": "Please share your account name and environment (prod, non prod or cgm prod)."}

    if sess["state"] == "awaiting_details":
        try:
            parts = text.split()
            info = AccountInfo(account_name=parts[0], environment=" ".join(parts[1:]))
        except (IndexError, ValidationError):
            raise HTTPException(status_code=400, detail="Invalid format. Use: <accountName> <environment>")

        raw = await fetch_account_details(info.account_name, info.environment)
        sess.update(state="details_fetched", data=raw)
        # AI-enhanced summary
        summary_prompt = (
            "You are an assistant. Summarize this account data for user: "
            f"{json.dumps(raw)}"
        )
        summary = client.get_chat_completions(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role":"user","content":summary_prompt}]
        ).choices[0].message.content
        return {"reply": summary + "\nWhat specific detail would you like?"}

    # Specific detail (IP)
    if sess["state"] == "details_fetched":
        if ip_matcher(doc) or re.search(r"\bips?\b", text, re.I):
            ips = sess["data"].get("ips", [])
            ip_prompt = ("Format these IPs as a user-friendly list: " + str(ips))
            ip_list = client.get_chat_completions(
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role":"user","content":ip_prompt}]
            ).choices[0].message.content
            return {"reply": ip_list}
        return {"reply": "Sorry, I can only fetch IPs from the account data."}

    # General Q&A
    resp = client.get_chat_completions(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role":"user","content": text}]
    )
    return {"reply": resp.choices[0].message.content}

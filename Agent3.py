from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum
import openai
import uvicorn
import os
import json
import re
import spacy

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Replace with your Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., 'gpt-4'

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = 'azure'
openai.api_version = '2023-07-01-preview'

# FastAPI app
app = FastAPI()

# Supported environments and intents
class Environment(str, Enum):
    prod = "prod"
    non_prod = "non prod"

class Intent(str, Enum):
    account_config = "account configuration search"
    file_tracking = "file tracking"
    jira_search = "issue search in jira"
    general = "general query"

# Memory store
conversation_memory: Dict[str, Dict] = {}

# Request model
class UserRequest(BaseModel):
    session_id: str
    user_input: str

# Number of past turns to include for context
MAX_CONTEXT_TURNS = 3

# NLP helper for parameter extraction
def extract_parameters_with_nlp(user_input: str) -> Dict[str, str]:
    doc = nlp(user_input.lower())
    params = {}
    
    # Extract environment
    if "non prod" in user_input.lower():
        params["environment"] = "non prod"
    elif "prod" in user_input.lower():
        params["environment"] = "prod"

    # Extract account names and proper nouns
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "PERSON"]:
            params.setdefault("account_name", ent.text)

    return params

# Helper: Call OpenAI LLM to extract intent and parameters
def analyze_user_input(user_input: str, history: List[Dict[str, str]]) -> Dict:
    system_prompt = (
        "You are an intelligent assistant that extracts intents and parameters from user input.\n"
        "Available intents: 'account configuration search', 'file tracking', 'issue search in jira'.\n"
        "Each intent may require specific parameters.\n"
        "For 'account configuration search' and 'file tracking', extract 'account_name' and 'environment' (prod or non prod).\n"
        "For 'issue search in jira', extract a clean 'jira_query'.\n"
        "If the input doesn't match any intent, set intent to 'general query'.\n"
        "Return a JSON object like: {\"intent\": ..., \"parameters\": {...}}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    for turn in history[-MAX_CONTEXT_TURNS:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        deployment_id=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    content = response["choices"][0]["message"]["content"]
    llm_result = json.loads(content)

    # Enrich LLM-extracted parameters with NLP
    if llm_result.get("intent") in [Intent.account_config, Intent.file_tracking]:
        nlp_params = extract_parameters_with_nlp(user_input)
        llm_result["parameters"].update({k: v for k, v in nlp_params.items() if k not in llm_result["parameters"]})

    return llm_result

# Dummy business logic implementations
def handle_account_config(account_name: str, environment: str):
    return f"Config data for account '{account_name}' in '{environment}' environment."

def handle_file_tracking(account_name: str, environment: str):
    return f"Tracking files for account '{account_name}' in '{environment}' environment."

def handle_jira_search(jira_query: str):
    return f"Searching Jira with query: '{jira_query}'"

# General fallback intent using Azure OpenAI
def handle_general_query(user_input: str, history: List[Dict[str, str]]) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    for turn in history[-MAX_CONTEXT_TURNS:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        deployment_id=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"]

# Main endpoint
@app.post("/chat")
def handle_chat(request: UserRequest):
    session = conversation_memory.setdefault(request.session_id, {
        "intent": None,
        "parameters": {},
        "history": []
    })

    analysis = analyze_user_input(request.user_input, session["history"])
    new_intent = analysis.get("intent", Intent.general)
    new_parameters = analysis.get("parameters", {})

    if new_intent != session.get("intent") and new_intent != Intent.general:
        session["parameters"] = new_parameters
        session["intent"] = new_intent
    else:
        session["parameters"].update(new_parameters)
        session["intent"] = session.get("intent") or new_intent

    intent = session["intent"]
    parameters = session["parameters"]

    if intent == Intent.account_config:
        missing = [p for p in ["account_name", "environment"] if p not in parameters]
        if missing:
            msg = f"Please provide: {', '.join(missing)}."
        else:
            msg = handle_account_config(parameters["account_name"], parameters["environment"])

    elif intent == Intent.file_tracking:
        missing = [p for p in ["account_name", "environment"] if p not in parameters]
        if missing:
            msg = f"Please provide: {', '.join(missing)}."
        else:
            msg = handle_file_tracking(parameters["account_name"], parameters["environment"])

    elif intent == Intent.jira_search:
        if "jira_query" not in parameters:
            msg = "Please provide a Jira search query."
        else:
            msg = handle_jira_search(parameters["jira_query"])

    else:
        msg = handle_general_query(request.user_input, session["history"])

    session["history"].append({
        "user": request.user_input,
        "bot": msg
    })

    return {"response": msg, "history": session["history"]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

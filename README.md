# Azure OpenAI FastAPI Agent

This project offers a chat agent that:

1. Uses **spaCy NLP** for intent classification and entity matching.
2. Leverages **Azure OpenAI** for both intent refinement and intelligent handling of backend API responses.
3. Manages a simple account‑details flow with dynamic follow‑ups.

## Features
- **Intent Tagging**: Hybrid spaCy matcher + Azure OpenAI classification for robust intent detection.
- **API Response Handling**: Raw backend JSON is post‑processed via Azure OpenAI to produce user‑friendly summaries or answers.
- **Validation**: Ensures account names are alphanumeric and environment is one of [prod, non prod, cgm prod].
- **General Q&A**: Any outside‑intent query is handled via Azure OpenAI chat.

## Setup
1. Clone & enter the repo.
2. Create `.env`:
   ```ini
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
   AZURE_OPENAI_API_KEY=<YOUR_API_KEY>
   AZURE_OPENAI_DEPLOYMENT=<DEPLOYMENT_NAME>
   ```
3. Install deps & NLP model:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Run server:
   ```bash
   uvicorn main:app --reload
   ```
5. Test via Postman on `POST /chat` with JSON `{ "session_id":"1","message":"..." }`.

# Deploy publicly (Streamlit Cloud)

Same approach as a typical campus demo: free hosting at `*.streamlit.app`.

## 1. Push this project to GitHub

Create a new public repository (example name: `meridian-procurement-intelligence`), then:

```powershell
cd C:\Users\user\Downloads\supply-chain-rag
git add .
git commit -m "Add Meridian procurement RAG assistant for HCL x Economic Times"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/meridian-procurement-intelligence.git
git push -u origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub
3. Click **New app**
4. Select your repo · branch `main` · main file `app.py`
5. Under **Advanced settings → Secrets**, paste:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your_gemini_key"
```

(Campus-approved alternative to OpenAI. Free key: https://aistudio.google.com/apikey)
6. Click **Deploy**

Your public link will look like:

`https://meridian-procurement-intelligence-xxxxx.streamlit.app`

Share that link with anyone — no localhost needed.

## 3. What visitors see

- Type a question (or tap an example)
- Get a grounded answer + document/page sources
- Knowledge base indexes automatically on first run

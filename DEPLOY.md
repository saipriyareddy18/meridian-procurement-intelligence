# Deploy so EVERYONE can open it (any phone/laptop)

This uses free **Streamlit Community Cloud** → a public link like:
`https://….streamlit.app`

---

## Step 1 — Create the GitHub repo (2 minutes)

1. Sign in as **saipriyareddy18**
2. Open: https://github.com/new
3. Repository name: `meridian-procurement-intelligence`
4. Public
5. **Do NOT** tick “Add a README”
6. Click **Create repository**

Then tell me “repo created” — I will push the code.

Or push yourself after creating:

```powershell
cd C:\Users\user\Downloads\supply-chain-rag
git push -u origin main
```

---

## Step 2 — Deploy on Streamlit Cloud

1. Open https://share.streamlit.io/ and sign in with **GitHub**
2. **New app**
3. Choose:
   - Repo: `saipriyareddy18/meridian-procurement-intelligence`
   - Branch: `main`
   - Main file: `app.py`
4. **Advanced settings → Secrets** — paste exactly:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "paste-your-gemini-key-here"
GOOGLE_API_KEY = "paste-your-gemini-key-here"
CHAT_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL = "models/gemini-embedding-001"
```

5. Click **Deploy**

Wait 1–3 minutes. Your public URL will appear.

Share that URL with anyone — works on any system, 24/7 (while Streamlit Cloud is up).

---

## Step 3 — First-time use on the live site

Visitors open the link → click **Index Documents** in the sidebar → ask questions.

(If indexing fails, confirm Secrets were saved, then reboot the app from Streamlit Cloud → Manage app → Reboot.)

import streamlit as st
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone
import hashlib
import numpy as np
import smtplib
from email.message import EmailMessage
import requests
from bs4 import BeautifulSoup
import json
import time
import re

# OpenAI client
from openai import OpenAI

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found — AI features will be disabled (search will produce simulated results).")

# -----------------------------
# Database init (uses boqs table now)
# -----------------------------
DB_FILE = "rfp_app_full.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS rfps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rfp_id TEXT UNIQUE,
        company TEXT,
        rfp_type TEXT,
        rfp_text TEXT,
        document TEXT,
        product TEXT,
        quantity INTEGER,
        deadline TEXT,
        location TEXT,
        status TEXT,
        proposal TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS skus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT UNIQUE,
        title TEXT,
        specs TEXT,
        cost REAL,
        lead_time_days INTEGER,
        embedding BLOB
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS boqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rfp_id TEXT,
        sku TEXT,
        qty INTEGER,
        notes TEXT,
        created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        po_id TEXT,
        rfp_id TEXT,
        vendor TEXT,
        total_amount REAL,
        status TEXT,
        created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS notices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rfp_id TEXT,
        event TEXT,
        details TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

# -----------------------------
# Seed SKUs if empty (unchanged)
# -----------------------------
def seed_skus():
    cur.execute("SELECT COUNT(*) FROM skus")
    if cur.fetchone()[0] == 0:
        sku_list = [
            ("CBL101","11kV XLPE Copper Cable","11kV, XLPE insulation, copper conductor", 9000.0, 14),
            ("CBL102","33kV PVC Aluminum Cable","33kV, PVC insulation, aluminum conductor", 15000.0, 21),
            ("CBL103","1.1kV PVC Copper Cable","1.1kV, PVC insulation, copper conductor", 2500.0, 7),
        ]
        for sku, title, specs, cost, lead in sku_list:
            cur.execute("INSERT OR IGNORE INTO skus (sku, title, specs, cost, lead_time_days, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                        (sku, title, specs, cost, lead, None))
        conn.commit()
seed_skus()

# -----------------------------
# Utility functions (unchanged except names for boqs)
# -----------------------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username, password):
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return bool(row and row[0] == hash_password(password))

def add_rfp(rec):
    cur.execute("""INSERT OR IGNORE INTO rfps (rfp_id, company, rfp_type, rfp_text, document, product, quantity, deadline, location, status, proposal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rec.get("rfp_id"), rec.get("company"), rec.get("rfp_type"), rec.get("rfp_text"),
                 rec.get("document"), rec.get("product"), rec.get("quantity"),
                 rec.get("deadline"), rec.get("location"), rec.get("status","Pending"), rec.get("proposal","")))
    conn.commit()

def list_rfps_df():
    df = pd.read_sql_query("SELECT * FROM rfps ORDER BY deadline", conn)
    return df

def update_proposal_and_status(rfp_id, proposal_text, status=None):
    if status:
        cur.execute("UPDATE rfps SET proposal = ?, status = ? WHERE rfp_id = ?", (proposal_text, status, rfp_id))
    else:
        cur.execute("UPDATE rfps SET proposal = ? WHERE rfp_id = ?", (proposal_text, rfp_id))
    conn.commit()

def insert_boq(rfp_id, sku, qty, notes=""):
    cur.execute("INSERT INTO boqs (rfp_id, sku, qty, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (rfp_id, sku, qty, notes, datetime.now(timezone.utc).isoformat()))
    conn.commit()

def list_boqs_for_rfp(rfp_id):
    df = pd.read_sql_query("SELECT * FROM boqs WHERE rfp_id = ? ORDER BY created_at DESC", conn, params=(rfp_id,))
    return df

def create_po(po_id, rfp_id, vendor, total_amount, status="Issued"):
    cur.execute("INSERT INTO pos (po_id, rfp_id, vendor, total_amount, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (po_id, rfp_id, vendor, total_amount, status, datetime.now(timezone.utc).isoformat()))
    conn.commit()

def list_pos():
    df = pd.read_sql_query("SELECT * FROM pos ORDER BY created_at DESC", conn)
    return df

def add_notice(rfp_id, event, details=""):
    cur.execute("INSERT INTO notices (rfp_id, event, details, timestamp) VALUES (?, ?, ?, ?)",
                (rfp_id, event, details, datetime.now(timezone.utc).isoformat()))
    conn.commit()

def list_notices(limit=200):
    df = pd.read_sql_query("SELECT * FROM notices ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    return df

# Embedding helpers (unchanged)
def get_embedding(text):
    if not client:
        return None
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        emb = resp.data[0].embedding
        return np.array(emb, dtype=np.float32)
    except Exception:
        return None

def ensure_sku_embeddings():
    cur.execute("SELECT id, sku, title, specs, embedding FROM skus")
    rows = cur.fetchall()
    for r in rows:
        sku_id, sku, title, specs, emb_blob = r
        if emb_blob is None:
            text = f"{sku} {title} {specs}"
            emb = get_embedding(text)
            if emb is not None:
                cur.execute("UPDATE skus SET embedding = ? WHERE id = ?", (emb.tobytes(), sku_id))
    conn.commit()

def load_skus_with_embeddings():
    cur.execute("SELECT sku, title, specs, cost, lead_time_days, embedding FROM skus")
    rows = cur.fetchall()
    out = []
    for sku, title, specs, cost, lead, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None
        out.append({"sku": sku, "title": title, "specs": specs, "cost": cost, "lead": lead, "emb": emb})
    return out

def send_email_smtp(to_email, subject, body, from_email=None):
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP not configured in .env.")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email or SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_SERVER, int(SMTP_PORT) if SMTP_PORT else 465) as smtp:
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)

# -----------------------------
# Search-like Scouting Agent
# -----------------------------
def fetch_page_text(url, timeout=10):
    try:
        headers = {"User-Agent":"Mozilla/5.0 (RFP-Scout/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script","style","noscript"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        pdfs = []
        for a in soup.find_all("a", href=True):
            href = a['href']
            if isinstance(href, str) and href.lower().endswith(".pdf"):
                pdfs.append(href if href.startswith("http") else requests.compat.urljoin(url, href))
        return text, pdfs
    except Exception:
        return None, []

def extract_rfps_via_ai(page_text, page_url):
    if not client:
        return []
    prompt = f"""
You are a Scouting Agent. From the webpage text below, extract RFP/tender entries as a JSON array.
Each item: rfp_id (generate if none), title, company, rfp_text, document_url, product, quantity, deadline (ISO or empty), location.
Return only a JSON array.
URL: {page_url}

Text:
{page_text}
"""
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=800)
        out = resp.choices[0].message.content
        start = out.find('[')
        end = out.rfind(']') + 1
        if start != -1 and end != -1:
            parsed = json.loads(out[start:end])
            if isinstance(parsed, list):
                return parsed
        parsed = json.loads(out)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

def search_like_engine_via_ai(query, n=3):
    """
    If OpenAI key available, ask it to behave like a search engine and return JSON array of candidate RFPs.
    If not, return simulated results.
    """
    if client:
        prompt = f"""
You are a search assistant specialized in Indian infrastructure tenders for cables/wires.
Given the user query: "{query}", return a JSON array of up to {n} candidate RFP items relevant to that query.
Each item: rfp_id, title, company, rfp_text, product, quantity, deadline (ISO or empty), location, document_url (optional).
Return ONLY a JSON array.
"""
        try:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=800)
            out = resp.choices[0].message.content
            start = out.find('[')
            end = out.rfind(']') + 1
            if start != -1 and end != -1:
                parsed = json.loads(out[start:end])
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            pass
    # fallback simulated
    results = []
    for i in range(n):
        results.append({
            "rfp_id": f"SRCH-{int(time.time())%100000 + i}",
            "title": f"Tender: supply of {query} - Lot {i+1}",
            "company": f"Government Dept {100+i}",
            "rfp_text": f"Requirement for supply of {query}. Full specs to be provided in PDF. Qty: {10*(i+1)}",
            "product": query,
            "quantity": 10*(i+1),
            "deadline": (datetime.now(timezone.utc).date()).isoformat(),
            "location": "India",
            "document_url": ""
        })
    return results

# -----------------------------
# Streamlit UI & navigation
# -----------------------------
st.set_page_config(page_title="Agentic RFP (updated)", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "Profile"

# Authentication (unchanged, no experimental_rerun)
if not st.session_state.logged_in:
    st.title("🔐 Sign In / Create Account")
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])
    with tab1:
        uname = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Sign In"):
            if verify_user(uname, pw):
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.success(f"Welcome {uname} — you are now signed in.")
                # leave page as Profile by default
                st.stop()
            else:
                st.error("Invalid credentials")
    with tab2:
        new_uname = st.text_input("New username", key="signup_user")
        new_pw = st.text_input("New password", type="password", key="signup_pwd")
        if st.button("Create Account"):
            if create_user(new_uname, new_pw):
                st.success("Account created — sign in now.")
            else:
                st.error("Username already exists.")
    st.stop()

# Sidebar with circle icon + username + (empty role line)
st.sidebar.markdown(
    f"""
    <div style='display:flex; align-items:center; gap:10px; padding:8px 6px;'>
      <div style='width:48px; height:48px; background:linear-gradient(135deg,#8B5CF6,#EC4899); border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:700;'>
        {st.session_state.username[:1].upper() if st.session_state.username else "U"}
      </div>
      <div>
        <div style='font-weight:700'>{st.session_state.username}</div>
        <div style='font-size:12px; color:#666;'> </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Menu (Profile first) - labels per your request
menu = st.sidebar.radio("Menu", [
    "Profile",
    "Dashboard",
    "Chatbox",
    "Find / Discover RFPs",
    "RFP Records",
    "Run AI Agents",
    "BOQ & POs",
    "Draft Quotation",
    "Notices",
    "Logout"
])

# ---------- Pages ----------

# Profile
if menu == "Profile":
    st.title("👤 Profile")
    st.write("Logged in as:", st.session_state.username)
    st.markdown("**Administrator access:** (future) — currently all users behave as standard user.")
    if st.button("Show proposals with content"):
        df = list_rfps_df()
        if df.empty:
            st.info("No RFPs available.")
        else:
            st.dataframe(df[["rfp_id","company","rfp_text","proposal"]])

# Chatbox
elif menu == "Chatbox":
    st.set_page_config(page_title="AI Chatbox", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
    .chat-container {
        max-width: 100%;
        margin: 0 auto;
        padding: 20px;
    }
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f1f1f1;
        color: black;
        margin-right: auto;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background-color: white;
        border-top: 1px solid #ddd;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("💬 AI Chatbox - Powered by OpenAI")
    st.markdown("Chat with an advanced AI assistant similar to ChatGPT. Query RFPs, update statuses, add notices, and more. The AI uses tools for database interactions.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input_disabled" not in st.session_state:
        st.session_state.chat_input_disabled = False

    # Display chat history with enhanced styling
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(f'<div class="chat-message {'user-message' if msg["role"] == "user" else 'assistant-message'}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Placeholder for current assistant response
    current_response_placeholder = st.empty()

    # Input with enhanced UI
    with st.container():
        col1, col2 = st.columns([1, 0.1])
        with col1:
            user_input = st.chat_input("Type your message here...", disabled=st.session_state.chat_input_disabled)
        with col2:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_input_disabled = True
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f'<div class="chat-message user-message">{user_input}</div>', unsafe_allow_html=True)
        st.session_state.generate_response = True
        st.rerun()

    # Generate response if flagged
    if st.session_state.get("generate_response", False):
        st.session_state.generate_response = False

        # Generate response with streaming
        if client:
            # Enhanced system prompt for ChatGPT-like behavior
            system_prompt = """You are a helpful AI assistant like ChatGPT, specialized in managing RFPs for cables and wires. Be conversational, clear, and proactive. Use tools when needed for database actions. Always confirm actions before executing if uncertain. Respond in a friendly, professional tone."""

            # Define tools (same as before)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "list_rfps",
                        "description": "List all RFPs in the database",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_rfp_status",
                        "description": "Update the status of an RFP",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "rfp_id": {"type": "string", "description": "The RFP ID"},
                                "status": {"type": "string", "description": "The new status"}
                            },
                            "required": ["rfp_id", "status"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "add_notice",
                        "description": "Add a notice for an RFP",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "rfp_id": {"type": "string", "description": "The RFP ID"},
                                "event": {"type": "string", "description": "The event"},
                                "details": {"type": "string", "description": "Details"}
                            },
                            "required": ["rfp_id", "event"]
                        }
                    }
                }
            ]

            try:
                # Streaming call to OpenAI
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}] + st.session_state.chat_history,
                    tools=tools,
                    tool_choice="auto",
                    stream=True
                )

                full_response = ""
                with current_response_placeholder.container():
                    with st.chat_message("assistant", avatar="🤖"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown('<div class="chat-message assistant-message">🤖 AI is thinking...</div>', unsafe_allow_html=True)
                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(f'<div class="chat-message assistant-message">{full_response}</div>', unsafe_allow_html=True)

                # Handle tool calls after streaming (always re-call non-streaming to check for tools)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": system_prompt}] + st.session_state.chat_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                    message = response.choices[0].message
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)

                            if function_name == "list_rfps":
                                df = list_rfps_df()
                                if df.empty:
                                    tool_result = "No RFPs found."
                                else:
                                    tool_result = df[['rfp_id', 'company', 'product', 'quantity', 'status']].head(10).to_string(index=False)
                            elif function_name == "update_rfp_status":
                                rfp_id = function_args["rfp_id"]
                                status = function_args["status"]
                                cur.execute("UPDATE rfps SET status = ? WHERE rfp_id = ?", (status, rfp_id))
                                conn.commit()
                                add_notice(rfp_id, "Status Updated", f"Status set to {status} by AI")
                                tool_result = f"✅ Updated RFP {rfp_id} status to {status}"
                            elif function_name == "add_notice":
                                rfp_id = function_args["rfp_id"]
                                event = function_args["event"]
                                details = function_args.get("details", "")
                                add_notice(rfp_id, event, details)
                                tool_result = f"✅ Added notice for RFP {rfp_id}: {event}"
                            else:
                                tool_result = "Unknown function called."

                            full_response += f"\n\n{tool_result}"
                            message_placeholder.markdown(f'<div class="chat-message assistant-message">{full_response}</div>', unsafe_allow_html=True)
                except Exception as e:
                    error_msg = f"Error in tool handling: {str(e)}"
                    full_response += f"\n\n{error_msg}"
                    message_placeholder.markdown(f'<div class="chat-message assistant-message">{full_response}</div>', unsafe_allow_html=True)

                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Error: Failed to get AI response. {str(e)}. Please check your API key or try again."
                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f'<div class="chat-message assistant-message">{error_msg}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        else:
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown('<div class="chat-message assistant-message">AI not available. Please check your OpenAI API key.</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": "AI not available. Please check your OpenAI API key."})

        st.session_state.chat_input_disabled = False
        st.rerun()

# Find / Discover RFPs (Search first, then Manual Entry)
elif menu == "Find / Discover RFPs":
    st.title("🔎 Find / Discover RFPs")
    st.markdown("First try a search-like discovery (AI-assisted). If you have exact target URLs, paste them below for scraping/scouting. Otherwise just enter keywords and click Search.")

    # Search-like input
    query = st.text_input("Search keywords (e.g., '11kV XLPE cable tender India')", "")
    urls_input = st.text_area("Optional: target URLs to scan (one per line). If left empty, the AI search-like engine will be used.", height=120)
    pages_to_scan = st.number_input("Max pages per URL", min_value=1, max_value=3, value=1)
    if st.button("Search / Scout"):
        found = []
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        if urls:
            st.info(f"Scanning {len(urls)} URLs (up to {pages_to_scan} pages each)...")
            for url in urls:
                for p in range(1, pages_to_scan+1):
                    scan_url = url if p == 1 else f"{url}?page={p}"
                    text, pdfs = fetch_page_text(scan_url)
                    if not text:
                        st.warning(f"Failed to fetch {scan_url}")
                        continue
                    # try AI extraction if available
                    recs = extract_rfps_via_ai(text, scan_url) if client else []
                    # if no AI or AI returned empty, make heuristic item
                    if not recs:
                        # naive heuristics: find lines containing 'tender' or 'supply'
                        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip())>30]
                        snippet = lines[0] if lines else ""
                        recs = [{
                            "rfp_id": f"SCOUT-{int(time.time())%100000}",
                            "title": snippet[:120],
                            "company": "",
                            "rfp_text": snippet,
                            "document_url": pdfs[0] if pdfs else "",
                            "product": query or "",
                            "quantity": 0,
                            "deadline": "",
                            "location": "",
                        }]
                    for r in recs:
                        r["_source_url"] = scan_url
                        found.append(r)
            if not found:
                st.info("No items found from provided URLs.")
            else:
                st.success(f"Found {len(found)} items from URLs (preview below).")
                preview = pd.DataFrame(found)
                st.dataframe(preview[["rfp_id","title","company","product","quantity","_source_url"]].fillna(""))
                to_import = st.multiselect("Select RFP ids to import to DB", preview["rfp_id"].tolist())
                if st.button("Import Selected to DB"):
                    for rid in to_import:
                        rec = preview[preview["rfp_id"]==rid].iloc[0].to_dict()
                        add_rfp({
                            "rfp_id": rec.get("rfp_id"),
                            "company": rec.get("company",""),
                            "rfp_type": "Scouted",
                            "rfp_text": rec.get("rfp_text",""),
                            "document": rec.get("document_url",""),
                            "product": rec.get("product",""),
                            "quantity": int(rec.get("quantity") or 0),
                            "deadline": rec.get("deadline") or "",
                            "location": rec.get("location",""),
                            "status": "Pending",
                            "proposal": ""
                        })
                        add_notice(rec.get("rfp_id"), "Imported (scout)", f"Imported by {st.session_state.username}")
                    st.success("Imported selected records to DB.")
        else:
            if not query.strip():
                st.warning("Enter keywords or paste URLs; nothing to search.")
            else:
                st.info("No URLs provided — using search-like AI to produce candidate RFPs.")
                candidates = search_like_engine_via_ai(query, n=3)
                if not candidates:
                    st.info("No candidates produced.")
                else:
                    st.success(f"Found {len(candidates)} candidate RFPs (preview).")
                    cand_df = pd.DataFrame(candidates)
                    st.dataframe(cand_df[["rfp_id","title","company","product","quantity","deadline"]].fillna(""))
                    col1, col2 = st.columns(2)
                    with col1:
                        to_import = st.multiselect("Select candidate RFP ids to import to DB", cand_df["rfp_id"].tolist())
                        if st.button("Import Selected Candidates"):
                            for rid in to_import:
                                rec = cand_df[cand_df["rfp_id"]==rid].iloc[0].to_dict()
                                add_rfp({
                                    "rfp_id": rec.get("rfp_id"),
                                    "company": rec.get("company",""),
                                    "rfp_type": "SearchCandidate",
                                    "rfp_text": rec.get("rfp_text",""),
                                    "document": rec.get("document_url",""),
                                    "product": rec.get("product",""),
                                    "quantity": int(rec.get("quantity") or 0),
                                    "deadline": rec.get("deadline") or "",
                                    "location": rec.get("location",""),
                                    "status":"Pending",
                                    "proposal":""
                                })
                                add_notice(rec.get("rfp_id"), "Imported (search)", f"Imported by {st.session_state.username}")
                            st.success("Imported selected search candidates to DB.")
                    with col2:
                        if st.button("Import All Candidates"):
                            for _, rec in cand_df.iterrows():
                                add_rfp({
                                    "rfp_id": rec.get("rfp_id"),
                                    "company": rec.get("company",""),
                                    "rfp_type": "SearchCandidate",
                                    "rfp_text": rec.get("rfp_text",""),
                                    "document": rec.get("document_url",""),
                                    "product": rec.get("product",""),
                                    "quantity": int(rec.get("quantity") or 0),
                                    "deadline": rec.get("deadline") or "",
                                    "location": rec.get("location",""),
                                    "status":"Pending",
                                    "proposal":""
                                })
                                add_notice(rec.get("rfp_id"), "Imported (search all)", f"Imported by {st.session_state.username}")
                            st.success(f"Imported all {len(cand_df)} search candidates to DB.")

    st.markdown("---")
    st.subheader("Manual RFP Entry (add any RFP yourself)")
    with st.form("manual_rfp_form"):
        rfp_id = st.text_input("RFP ID (e.g., RFP-12345)")
        company = st.text_input("Company")
        rfp_type = st.selectbox("RFP Type", ["Tender","Proposal","EOI","Other"])
        rfp_text = st.text_area("RFP Text / Description", height=160)
        document = st.text_input("Document filename (PDF stored in project folder) - optional")
        product = st.text_input("Product (e.g., 11kV XLPE Cable)")
        quantity = st.number_input("Quantity", min_value=0, value=0)
        deadline = st.date_input("Deadline")
        location = st.text_input("Location")
        submitted = st.form_submit_button("Save Manual RFP")
        if submitted:
            add_rfp({
                "rfp_id": rfp_id or f"MAN-{int(time.time())}",
                "company": company,
                "rfp_type": rfp_type,
                "rfp_text": rfp_text,
                "document": document,
                "product": product,
                "quantity": quantity,
                "deadline": deadline.isoformat(),
                "location": location,
                "status":"Pending",
                "proposal":""
            })
            add_notice(rfp_id or f"MAN-{int(time.time())}", "Manual Entry", f"Added by {st.session_state.username}")
            st.success("Manual RFP saved to DB.")

# RFP Records
# RFP Records
elif menu == "RFP Records":
    st.title("📂 RFP Records")
    df = list_rfps_df()
    if df.empty:
        st.info("No RFP records.")
    else:
        # Use rfp_text as the detailed description/title field in DB
        # Provide short snippet for display in table
        df = df.copy()
        if "rfp_text" not in df.columns:
            df["rfp_text"] = ""
        df["short_title"] = df["rfp_text"].fillna("").apply(lambda t: (t[:120] + "...") if len(t) > 120 else t)

        q_company = st.text_input("Filter by company")
        q_product = st.text_input("Filter by product")
        filtered = df.copy()
        if q_company:
            filtered = filtered[filtered["company"].str.contains(q_company, case=False, na=False)]
        if q_product:
            filtered = filtered[filtered["product"].str.contains(q_product, case=False, na=False)]

        # Show table using columns that exist in DB
        display_cols = ["rfp_id", "company", "short_title", "product", "quantity", "deadline", "status"]
        st.dataframe(filtered[display_cols].rename(columns={"rfp_id":"RFP_ID","company":"Company","short_title":"Title","product":"Product","quantity":"Quantity","deadline":"Deadline","status":"Status"}).fillna(""))

        sel = st.selectbox("Select an RFP ID for details", ["--none--"] + filtered["rfp_id"].tolist())
        if sel and sel != "--none--":
            r = df[df["rfp_id"]==sel].iloc[0]
            st.markdown(f"### {r.get('rfp_id')} — {r.get('company')}")
            # show a reasonable 'title' derived from rfp_text
            st.write("Title / Short description:", (r.get("rfp_text") or "")[:400])
            st.write("Product:", r.get("product"), "| Qty:", r.get("quantity"))
            st.write("Deadline:", r.get("deadline"))
            st.write("Location:", r.get("location"))
            if r.get("document") and os.path.exists(r.get("document")):
                with open(r.get("document"), "rb") as f:
                    st.download_button("Open / Download RFP PDF", data=f.read(), file_name=os.path.basename(r.get("document")), mime='application/pdf')
            st.markdown("**Proposal status**: " + (r.get("status") or ""))
            if st.button("Mark as Expired"):
                cur.execute("UPDATE rfps SET status = 'Expired' WHERE rfp_id = ?", (sel,))
                conn.commit()
                add_notice(sel, "Marked Expired", f"By {st.session_state.username}")
                st.success("Marked expired.")


# Run AI Agents
elif menu == "Run AI Agents":
    st.title("⚡ Run AI Agents (Sales → Technical → Pricing)")
    df = list_rfps_df()
    if df.empty:
        st.info("No RFPs available.")
    else:
        choices = st.multiselect("Select RFPs to process", df['rfp_id'].tolist())
        if st.button("Run AI Agents"):
            if not choices:
                st.warning("Select at least one RFP to run agents.")
            else:
                if client:
                    ensure_sku_embeddings()
                skus = load_skus_with_embeddings()

                for rfp_id in choices:
                    row = pd.read_sql_query("SELECT * FROM rfps WHERE rfp_id = ?", conn, params=(rfp_id,)).iloc[0]
                    st.markdown(f"### Processing RFP: {rfp_id} — {row['company']}")

                    # -----------------------
                    # Sales Summary (collapsible)
                    # -----------------------
                    with st.expander("📄 Sales Summary", expanded=True):
                        if client:
                            prompt_sales = f"Summarize the RFP in 4 bullets: product, qty, key specs, testing. Respond in plain text only, using simple bullet points without any LaTeX, math notation, or special formatting.\nRFP:\n{row['rfp_text']}"
                            try:
                                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_sales}])
                                summary = res.choices[0].message.content
                            except Exception as e:
                                summary = f"(Sales AI failed: {e})"
                        else:
                            summary = "AI key missing; no summary."
                        st.write(summary)

                    # -----------------------
                    # Technical Match (collapsible)
                    # -----------------------
                    with st.expander("🛠 Technical Match", expanded=False):
                        tech_output = "No match"
                        best = None
                        if client:
                            try:
                                emb = client.embeddings.create(model="text-embedding-3-small", input=summary if summary else row['rfp_text']).data[0].embedding
                                emb = np.array(emb, dtype=np.float32)
                                best_score = -1.0
                                for s in skus:
                                    if s['emb'] is None:
                                        continue
                                    sim = np.dot(emb, s['emb']) / (np.linalg.norm(emb)*np.linalg.norm(s['emb'])+1e-8)
                                    if sim > best_score:
                                        best_score = sim
                                        best = s
                                if best:
                                    tech_output = f"Best match: {best['sku']} — {best['title']} (score {best_score:.3f})\nSpecs: {best['specs']}\nLead time: {best['lead']} days\nUnit cost: ₹{best['cost']:.2f}"
                                else:
                                    tech_output = "No embedding match or embeddings missing."
                            except Exception as e:
                                # Fallback to AI-generated technical match
                                prompt_tech = f"Technical Agent: Based on the RFP summary '{summary}', suggest the best matching cable/wire SKU from our catalog: CBL101 (11kV XLPE Copper Cable, 11kV XLPE insulation, copper conductor, ₹9000, 14 days), CBL102 (33kV PVC Aluminum Cable, 33kV PVC insulation, aluminum conductor, ₹15000, 21 days), CBL103 (1.1kV PVC Copper Cable, 1.1kV PVC insulation, copper conductor, ₹2500, 7 days). Provide a description in this format: Best match: SKU — Title\nSpecs: ...\nLead time: ... days\nUnit cost: ₹.... Respond in plain text only, without LaTeX, math notation, or special formatting."
                                try:
                                    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_tech}])
                                    tech_output = res.choices[0].message.content
                                except Exception as e2:
                                    tech_output = f"Technical AI failed: {e2}"
                        else:
                            tech_output = "AI key missing; cannot match."
                        st.info(tech_output)

                        # Create BOQ
                        with st.expander("➕ Create BOQ from Technical Match"):
                            if best:
                                create_qty = st.number_input(f"Qty for SKU {best['sku']}", min_value=1, value=int(row['quantity']), key=f"boq_{rfp_id}")
                                bom_notes = st.text_input("Notes (optional)", key=f"boq_notes_{rfp_id}")
                                if st.button(f"Save BOQ for {rfp_id}", key=f"save_boq_{rfp_id}"):
                                    insert_boq(rfp_id, best['sku'], create_qty, bom_notes)
                                    add_notice(rfp_id, "BOQ Created", f"SKU {best['sku']} x{create_qty} by {st.session_state.username}")
                                    st.success("BOQ saved.")

                    # -----------------------
                    # Pricing (collapsible)
                    # -----------------------
                    with st.expander("💰 Pricing", expanded=False):
                        if client:
                            prompt_price = f"Pricing Agent: produce unit price, testing, freight, taxes assumptions, total in INR. Use simple numbers and text for calculations, no LaTeX or math notation. Structure as a clear breakdown with totals. Respond in plain text only.\nSummary:\n{summary}\nTech Match:\n{tech_output}"
                            try:
                                res_p = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_price}])
                                pricing = res_p.choices[0].message.content
                            except Exception as e:
                                pricing = f"(Pricing AI failed: {e})"
                        else:
                            pricing = "AI key missing."

                        st.warning(pricing)

                    # -----------------------
                    # Summary (collapsible)
                    # -----------------------
                    with st.expander("📋 Summary", expanded=False):
                        if client:
                            prompt_summary = f"Summarize the entire proposal in 3-4 sentences. Respond in plain text only, without LaTeX, math notation, or special formatting.\nSales Summary: {summary}\nTechnical Match: {tech_output}\nPricing: {pricing}"
                            try:
                                res_s = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt_summary}])
                                summary_text = res_s.choices[0].message.content
                            except Exception as e:
                                summary_text = f"(Summary AI failed: {e})"
                        else:
                            summary_text = "AI key missing; no summary."
                        st.write(summary_text)

                    # -----------------------
                    # Save combined proposal
                    # -----------------------
                    combined = f"SALES:\n{summary}\n\nTECH:\n{tech_output}\n\nPRICING:\n{pricing}"
                    update_proposal_and_status(rfp_id, combined, status="Reviewed")
                    add_notice(rfp_id, "Proposal Generated", f"Generated by AI for {rfp_id}")
                    st.success(f"Saved proposal for {rfp_id} and status=Reviewed.")
# BOQ & POs
elif menu == "BOQ & POs":
    pass  # Placeholder to ensure the block is not empty
    pass  # Placeholder to ensure the block is not empty
    st.title("🧾 BOQ & Purchase Orders")
    st.subheader("View / Create BOQ entries")
    all_rfps = pd.read_sql_query("SELECT rfp_id FROM rfps", conn)
    sel = st.selectbox("Choose RFP", ["--select--"] + all_rfps['rfp_id'].tolist())
    if sel and sel != "--select--":
        bdf = list_boqs_for_rfp(sel)
        if bdf.empty:
            st.info("No BOQ entries.")
        else:
            st.dataframe(bdf[['sku','qty','notes','created_at']].rename(columns={'sku':'SKU','qty':'Qty','notes':'Notes','created_at':'Created'}))
        st.markdown("Add a BOQ line")
        skus = load_skus_with_embeddings()
        sku_choice = st.selectbox("SKU", ["--select--"] + [s['sku'] for s in skus])
        qty = st.number_input("Quantity", min_value=1, value=1)
        notes = st.text_input("Notes")
        if st.button("Add BOQ Line"):
            if sku_choice and sku_choice != "--select--":
                insert_boq(sel, sku_choice, qty, notes)
                add_notice(sel, "BOQ Added", f"{sku_choice} x{qty} by {st.session_state.username}")
                st.success("BOQ line saved.")
            else:
                st.error("Select a SKU.")

    st.markdown("---")
    st.subheader("Create PO")
    df_rfps = pd.read_sql_query("SELECT rfp_id, company FROM rfps", conn)
    sel2 = st.selectbox("Select RFP for PO", ["--select--"] + df_rfps['rfp_id'].tolist(), key="po_rfp")
    if sel2 and sel2 != "--select--":
        r_info = df_rfps[df_rfps['rfp_id']==sel2].iloc[0]
        po_id = st.text_input("PO ID")
        vendor = st.text_input("Vendor", value=r_info['company'])
        total = st.number_input("Estimated total (INR)", min_value=0.0)
        if st.button("Create PO"):
            create_po(po_id, sel2, vendor, float(total))
            add_notice(sel2, "PO Created", f"PO {po_id} created by {st.session_state.username}")
            st.success(f"PO {po_id} created.")

    st.markdown("---")
    st.subheader("Recent POs")
    p_df = list_pos()
    if p_df.empty:
        st.info("No POs yet.")
    else:
        st.dataframe(p_df[['po_id','rfp_id','vendor','total_amount','status','created_at']])

# Draft Quotation / Email
elif menu == "Draft Quotation":
    st.title("✉️ Draft Quotation & Export PDF")
    df = list_rfps_df()
    if df.empty:
        st.info("No RFPs.")
    else:
        selected = st.selectbox("Choose RFP", df['rfp_id'].tolist())
        rec = df[df['rfp_id'] == selected].iloc[0]
        st.markdown(f"### {selected} — {rec['company']}")
        current = rec['proposal'] if rec['proposal'] else ""
        edited = st.text_area("Proposal / Email body (editable)", value=current, height=320)

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Save Proposal (mark Reviewed)"):
                update_proposal_and_status(selected, edited, status="Reviewed")
                add_notice(selected, "Proposal Saved", f"Saved by {st.session_state.username}")
                st.success("Proposal saved.")
        with col2:
            to_email = st.text_input("Recipient Email (buyer's contact)", key="to_email")
            subject = st.text_input("Email subject", value=f"Proposal for {selected} — {rec['company']}", key="email_subject")
            if st.button("Send Email via SMTP"):
                if not SMTP_SERVER or not SMTP_USER or not SMTP_PASS:
                    st.error("SMTP not configured in .env.")
                else:
                    try:
                        send_email_smtp(to_email, subject, edited)
                        update_proposal_and_status(selected, edited, status="Proposal Sent")
                        add_notice(selected, "Proposal Sent via Email", f"Sent to {to_email} by {st.session_state.username}")
                        st.success("Email sent and status updated.")
                    except Exception as e:
                        st.error(f"Email send failed: {e}")
        with col3:
            def build_proposal_html(rfp_rec, proposal_text):
                bom_df = list_boqs_for_rfp(rfp_rec['rfp_id'])
                css = """<style>body{font-family:Arial;color:#222} table,th,td{border:1px solid #ddd;border-collapse:collapse;padding:8px} th{background:#f5f5f5}</style>"""
                meta = f"<h1>Your Company</h1><div><strong>Proposal for: </strong>{rfp_rec['company']}</div><div><strong>RFP:</strong> {rfp_rec['rfp_text']}</div>"
                bom_html = ""
                if not bom_df.empty:
                    bom_html += "<h3>BOQ</h3><table><tr><th>SKU</th><th>Qty</th><th>Notes</th></tr>"
                    for _, r in bom_df.iterrows():
                        bom_html += f"<tr><td>{r['sku']}</td><td>{r['qty']}</td><td>{r['notes'] or ''}</td></tr>"
                    bom_html += "</table>"
                html = f"<!doctype html><html><head>{css}</head><body>{meta}<h3>Proposal</h3><div>{proposal_text.replace('\\n','<br/>')}</div>{bom_html}<div style='margin-top:18px'><small>Generated on {datetime.now(timezone.utc).isoformat()}</small></div></body></html>"
                return html

            if st.button("Export to PDF"):
                html = build_proposal_html(rec, edited)
                # try weasy/pdfkit/reportlab fallback (same as before)
                try:
                    from weasyprint import HTML
                    pdf_bytes = HTML(string=html).write_pdf()
                except Exception:
                    try:
                        import pdfkit
                        path = f"proposals/{selected}_proposal.pdf"
                        os.makedirs("proposals", exist_ok=True)
                        pdfkit.from_string(html, path)
                        with open(path, "rb") as f:
                            pdf_bytes = f.read()
                    except Exception:
                        from reportlab.lib.pagesizes import A4
                        from reportlab.pdfgen import canvas
                        from io import BytesIO
                        buffer = BytesIO()
                        c = canvas.Canvas(buffer, pagesize=A4)
                        textobject = c.beginText(40, 800)
                        plain = re.sub('<[^<]+?>', '', html)
                        for ln in plain.splitlines():
                            if textobject.getY() < 40:
                                c.drawText(textobject)
                                c.showPage()
                                textobject = c.beginText(40, 800)
                            textobject.textLine(ln[:200])
                        c.drawText(textobject)
                        c.save()
                        pdf_bytes = buffer.getvalue()
                        buffer.close()
                # save & download
                os.makedirs("proposals", exist_ok=True)
                out_path = f"proposals/{selected}_proposal.pdf"
                with open(out_path, "wb") as f:
                    f.write(pdf_bytes)
                add_notice(selected, "Proposal Exported", f"Exported by {st.session_state.username}")
                st.success(f"PDF exported to {out_path}")
                st.download_button("Download Proposal PDF", data=pdf_bytes, file_name=os.path.basename(out_path), mime="application/pdf")

# Notices
elif menu == "Notices":
    st.title("🔔 Notices / Audit Trail")
    n = list_notices()
    if n.empty:
        st.info("No notices yet.")
    else:
        st.dataframe(n.rename(columns={'rfp_id':'RFP_ID','event':'Event','details':'Details','timestamp':'Timestamp'}))

# Dashboard (kept but placed after main flow)
elif menu == "Dashboard":
    st.title("📊 Dashboard")
    df = list_rfps_df()
    if df.empty:
        st.info("No RFPs yet.")
    else:
        today = pd.to_datetime(datetime.today().date())
        # mark expired
        try:
            df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
            expired_ids = df[df['deadline'] < today]['rfp_id'].tolist()
            for rid in expired_ids:
                cur.execute("UPDATE rfps SET status = 'Expired' WHERE rfp_id = ?", (rid,))
            conn.commit()
        except Exception:
            pass
        st.dataframe(list_rfps_df()[['rfp_id','company','product','quantity','deadline','status']])

# Logout (no experimental rerun)
elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out — refresh page to sign in again.")
    st.stop()

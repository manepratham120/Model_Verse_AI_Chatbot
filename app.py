import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os

load_dotenv()

st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] * {
        color: #e8e4ff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        color: #b8b0e8 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
        border-color: rgba(138,99,255,0.7) !important;
    }

    /* Sidebar title */
    .sidebar-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.45rem;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: #8880aa !important;
        margin-bottom: 1.6rem;
    }

    /* ── Provider badge ── */
    .provider-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    .badge-groq    { background: rgba(249,115,22,0.18); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
    .badge-openai  { background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .badge-gemini  { background: rgba(59,130,246,0.18); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .badge-ollama  { background: rgba(168,85,247,0.18); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }

    /* ── Clear button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, rgba(138,99,255,0.25), rgba(96,165,250,0.2));
        color: #c4b5fd !important;
        border: 1px solid rgba(138,99,255,0.4) !important;
        border-radius: 12px !important;
        padding: 0.55rem 1rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
        margin-top: 0.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(138,99,255,0.45), rgba(96,165,250,0.35)) !important;
        border-color: rgba(138,99,255,0.7) !important;
        color: #fff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 18px rgba(138,99,255,0.3);
    }

    /* ── Main area header ── */
    .main-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        background: linear-gradient(90deg, #a78bfa 10%, #60a5fa 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .main-tagline {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    /* ── Active model pill ── */
    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #a78bfa;
        margin-bottom: 1.2rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 14px !important;
        font-size: 0.95rem !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


MODEL_OPTIONS = {
    "Gemini": ["gemini-3.1-flash-lite-preview","gemini-2.5-flash"],
    "Groq":   ["llama-3.3-70b-versatile","gemma2-9b-it"],
    "Ollama": ["mistral", "gemma:2b"],
}

PROVIDER_ICONS = {
    "OpenAI": "🟢", "Gemini": "🔵", "Groq": "🟠", "Ollama": "🟣"
}

BADGE_CLASS = {
    "OpenAI": "badge-openai", "Gemini": "badge-gemini",
    "Groq": "badge-groq", "Ollama": "badge-ollama",
}



def get_llm(provider, model_name):
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=0.1,
                          api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "Groq":
        return ChatGroq(model=model_name, temperature=0.1,
                        api_key=os.getenv("GROQ_API_KEY"))
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.1,
                                      api_key=os.getenv("GEMINI_API_KEY"))
    elif provider == "Ollama":
        return ChatOllama(model=model_name, temperature=0.1)
    else:
        st.error("Unsupported provider selected.")
        return None


for key, val in [("provider", "Groq"),
                 ("model_name", MODEL_OPTIONS["Groq"][0]),
                 ("chat_history", []),
                 ("temperature", 0.5)]:
    if key not in st.session_state:
        st.session_state[key] = val



with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ ModelVerse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Multi-provider AI chat</div>', unsafe_allow_html=True)

    st.markdown("**🏢 Provider**")
    provider = st.selectbox(
        "Select Provider",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.provider),
        label_visibility="collapsed",
    )

    badge = BADGE_CLASS.get(provider, "badge-groq")
    icon  = PROVIDER_ICONS.get(provider, "🤖")
    st.markdown(
        f'<div><span class="provider-badge {badge}">{icon} {provider}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**🧠 Model**")
    default_idx = (
        MODEL_OPTIONS[provider].index(st.session_state.model_name)
        if provider == st.session_state.provider
           and st.session_state.model_name in MODEL_OPTIONS[provider]
        else 0
    )
    model_name = st.selectbox(
        "Select Model",
        options=MODEL_OPTIONS[provider],
        index=default_idx,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Stats
    msg_count = len(st.session_state.chat_history)
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    background:rgba(255,255,255,0.05);border-radius:10px;padding:10px 14px;
                    font-size:0.82rem;color:#b0a8d8;margin-bottom:1rem;">
            <span>💬 Messages</span>
            <span style="font-weight:600;color:#a78bfa;">{msg_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.73rem;color:#5c5480;text-align:center;line-height:1.5;">'
        'Powered by LangChain<br>& Streamlit</div>',
        unsafe_allow_html=True,
    )

st.session_state.provider   = provider
st.session_state.model_name = model_name


st.markdown('<div class="main-title">🚨  Generative AI Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-tagline">Ask anything — switch models anytime from the sidebar.</div>',
    unsafe_allow_html=True,
)

icon  = PROVIDER_ICONS.get(provider, "🤖")
st.markdown(
    f'<div class="model-pill">{icon} {provider} &nbsp;·&nbsp; {model_name}</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = st.chat_input("Ask anything…")

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    llm = get_llm(provider, model_name)
    if llm:
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = llm.invoke(
                    input=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        *st.session_state.chat_history,
                    ]
                )
                if isinstance(response.content, list):
                    assistant_response = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in response.content
                    )
                else:
                    assistant_response = response.content
                st.markdown(assistant_response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response}
        )
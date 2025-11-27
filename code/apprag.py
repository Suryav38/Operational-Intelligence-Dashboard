import streamlit as st
from datetime import timedelta, datetime
from pathlib import Path
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# RAG imports
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# ======================================
# 1. Configuration & Styling
# ======================================
st.set_page_config(
    page_title="Intelligent Operations AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=""
)

# Professional CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Modern KPI Card */
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 4px solid #3498db;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover { transform: translateY(-2px); }
    .metric-label { font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 5px 0; }
    .metric-sub { font-size: 0.75rem; color: #95a5a6; }
    
    /* Alert Boxes */
    .alert-box { padding: 10px 15px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem; border-left: 4px solid; }
    .alert-crit { background-color: #fdeaea; border-color: #e74c3c; color: #c0392b; }
    .alert-warn { background-color: #fff8e1; border-color: #f1c40f; color: #7f8c8d; }
    
    /* --- CHAT FIX (AGGRESSIVE) --- */
    /* Target the container and ALL children elements */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Force text color for user and assistant messages */
    div[data-testid="stChatMessage"] * {
        color: #1a1a1a !important;
    }
    
    /* Differentiate User vs Assistant slightly */
    div[data-testid="stChatMessage"][data-author="user"] {
        background-color: #e3f2fd !important; 
    }
    /* ----------------------------- */

</style>
""", unsafe_allow_html=True)

# ======================================
# 2. RAG Setup (Cached)
# ======================================
@st.cache_resource
def initialize_rag():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"
        load_dotenv(env_path)
        
        persist_dir = root / "data" / "manrag" / "chroma_store"
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Low temp for factual answers
        return retriever, llm, None
    except Exception as e:
        return None, None, str(e)

# ======================================
# 3. Data Loading & Advanced Logic
# ======================================
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@st.cache_data
def load_data():
    root = get_project_root()
    data_path = root / "data" / "manrag" / "structured"

    if not data_path.exists():
        st.error("❌ Data missing. Run `data_generation.py` first.")
        st.stop()

    prod = pd.read_csv(data_path / "production_daily.csv", parse_dates=["date"])
    dt = pd.read_csv(data_path / "downtime_events.csv", parse_dates=["start_time", "end_time"])
    machines = pd.read_csv(data_path / "dim_machine.csv")
    
    return prod, dt, machines

def calculate_advanced_metrics(prod_df, dt_df):
    """Calculates operational AND reliability metrics."""
    # Production
    total_produced = prod_df["produced_units"].sum()
    total_scrap = prod_df["scrap_units"].sum()
    good_units = max(0, total_produced - total_scrap)
    
    # Financials
    EST_SCRAP_COST = 15.0
    EST_DOWNTIME_COST_HR = 800.0
    scrap_loss = total_scrap * EST_SCRAP_COST
    
    # Time
    planned_hours = prod_df["planned_hours"].sum()
    runtime_hours = prod_df["runtime_hours"].sum()
    
    # Downtime
    if not dt_df.empty:
        total_dt_min = dt_df["duration_min"].sum()
        total_dt_hours = total_dt_min / 60.0
        breakdown_count = len(dt_df)
    else:
        total_dt_min = 0
        total_dt_hours = 0
        breakdown_count = 0

    downtime_loss = total_dt_hours * EST_DOWNTIME_COST_HR
    total_financial_loss = scrap_loss + downtime_loss

    # MTBF / MTTR
    if breakdown_count > 0:
        mtbf = runtime_hours / breakdown_count
        mttr = total_dt_min / breakdown_count 
    else:
        mtbf = runtime_hours 
        mttr = 0.0

    # OEE Components
    availability = (runtime_hours / planned_hours) if planned_hours > 0 else 0.0
    quality = (good_units / total_produced) if total_produced > 0 else 0.0
    
    if "standard_cycle_time_sec" in prod_df.columns:
        ideal_time_sec = (prod_df["produced_units"] * prod_df["standard_cycle_time_sec"]).sum()
        actual_time_sec = runtime_hours * 3600
        performance = (ideal_time_sec / actual_time_sec) if actual_time_sec > 0 else 0.0
        performance = min(1.0, performance)
    else:
        performance = 0.90

    oee = availability * performance * quality * 100

    return {
        "oee": round(oee, 1),
        "availability": round(availability * 100, 1),
        "performance": round(performance * 100, 1),
        "quality": round(quality * 100, 1),
        "scrap_rate": round((1-quality)*100, 2),
        "downtime_hours": round(total_dt_hours, 1),
        "mtbf": round(mtbf, 1),
        "mttr": round(mttr, 1),
        "financial_loss": int(total_financial_loss),
        "breakdown_count": breakdown_count
    }

# ======================================
# 4. Initialization
# ======================================
retriever, llm, rag_error = initialize_rag()
prod, dt, machines = load_data()

# ======================================
# 5. Sidebar Filters
# ======================================
st.sidebar.title("Filters")

line_opts = ["All"] + sorted(prod["line_id"].unique().tolist())
sel_line = st.sidebar.selectbox("Line", line_opts)

if sel_line == "All":
    machine_opts = ["All"] + sorted(machines["machine_id"].unique().tolist())
else:
    subset = machines[machines["line_id"] == sel_line]
    machine_opts = ["All"] + sorted(subset["machine_id"].unique().tolist())
sel_machine = st.sidebar.selectbox("Machine Asset", machine_opts)

# Date Filter
min_d, max_d = prod["date"].min().date(), prod["date"].max().date()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start", value=max_d - timedelta(days=30), min_value=min_d, max_value=max_d)
end_date = col_d2.date_input("End", value=max_d, min_value=min_d, max_value=max_d)

# ======================================
# 6. Data Filtering
# ======================================
p_mask = (prod["date"].dt.date >= start_date) & (prod["date"].dt.date <= end_date)
if sel_line != "All": p_mask &= (prod["line_id"] == sel_line)
prod_f = prod[p_mask].copy()

d_mask = (dt["start_time"].dt.date >= start_date) & (dt["start_time"].dt.date <= end_date)
if sel_line != "All": d_mask &= (dt["line_id"] == sel_line)
if sel_machine != "All": d_mask &= (dt["machine_id"] == sel_machine)
dt_f = dt[d_mask].copy()

kpis = calculate_advanced_metrics(prod_f, dt_f)

# ======================================
# 7. Dashboard Header
# ======================================
st.title("Operation Dashboard")
st.markdown(f"**Scope:** {sel_line} > {sel_machine} | **Range:** {start_date} to {end_date}")
st.markdown("---")

col_alert, col_spacer = st.columns([2, 1])
with col_alert:
    if kpis['oee'] < 65:
        st.markdown(f"<div class='alert-box alert-crit'>⚠️ **Critical OEE ({kpis['oee']}%)**: Production efficiency is critically low. Investigate Top 3 Downtime causes immediately.</div>", unsafe_allow_html=True)
    elif kpis['scrap_rate'] > 4:
         st.markdown(f"<div class='alert-box alert-warn'>🔸 **High Scrap Rate ({kpis['scrap_rate']}%)**: Quality yield is impacting financial margins.</div>", unsafe_allow_html=True)

# ======================================
# 8. High-Impact KPI Row
# ======================================
metric_cols = st.columns(5)
metrics_def = [
    ("OEE Score", f"{kpis['oee']}%", f"Perf: {kpis['performance']}%", "#3498db"),
    ("Availability", f"{kpis['availability']}%", f"DT: {kpis['downtime_hours']}h", "#2ecc71"),
    ("MT Between Failure", f"{kpis['mtbf']}h", "Target: >48h", "#9b59b6"),
    ("Mean Time To Repair", f"{kpis['mttr']}m", "Target: <30m", "#e67e22"),
    ("Est. Financial Loss", f"${kpis['financial_loss']:,}", "Scrap + DT Cost", "#e74c3c")
]

for col, (label, val, sub, color) in zip(metric_cols, metrics_def):
    col.markdown(f"""
    <div class="metric-container" style="border-top-color: {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color}">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") 

# ======================================
# 9. Analytics Tabs
# ======================================
tab_prod, tab_qual, tab_rel = st.tabs(["Production & Targets", "Quality & Yield (Granular)", "Reliability Analysis"])

# --- Tab 1: Production ---
with tab_prod:
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.markdown("##### Production Volume vs Daily Target")
        if not prod_f.empty:
            daily = prod_f.groupby("date")["produced_units"].sum().reset_index()
            
            # --- FIXED TARGET LOGIC ---
            # If All Lines selected: Target = 210,000 / day
            # If Specific Line selected: Target = 70,000 / day
            daily_target = 70000 if sel_line != "All" else 210000
            daily["target"] = daily_target
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily['date'], y=daily['produced_units'], name="Actual Output", marker_color="#3498db"))
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['target'], name=f"Target ({daily_target//1000}k)", line=dict(color="#e74c3c", width=3)))
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
    
    with col_p2:
        st.markdown("##### Performance Heatmap")
        if not prod_f.empty:
            daily_oee = prod_f.groupby("date")[["runtime_hours", "planned_hours"]].sum().reset_index()
            daily_oee["util"] = (daily_oee["runtime_hours"] / daily_oee["planned_hours"]) * 100
            fig_trend = px.line(daily_oee, x="date", y="util", title="Daily Utilization %", line_shape="spline")
            fig_trend.update_traces(line_color="#2ecc71")
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)

# --- Tab 2: Granular Quality ---
with tab_qual:
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown("##### Line-wise Quality Yield (%)")
        if not prod_f.empty:
            line_qual = prod_f.groupby("line_id").agg({
                "produced_units": "sum", 
                "scrap_units": "sum"
            }).reset_index()
            line_qual["Quality %"] = ((line_qual["produced_units"] - line_qual["scrap_units"]) / line_qual["produced_units"] * 100).round(1)
            
            fig_line_q = px.bar(line_qual, x="line_id", y="Quality %", color="Quality %", color_continuous_scale="Teal", range_y=[80, 100], text="Quality %")
            fig_line_q.update_layout(height=350)
            st.plotly_chart(fig_line_q, use_container_width=True)
            
    with col_q2:
        st.markdown("##### Machine-wise High Scrap (Top 10 Proxy)")
        # Using breakdown frequency as proxy for unstable machines
        if not dt_f.empty:
            machine_issues = dt_f.groupby("machine_id").size().reset_index(name="Issue Count")
            machine_issues = machine_issues.sort_values("Issue Count", ascending=False).head(10)
            
            fig_mach = px.bar(machine_issues, x="machine_id", y="Issue Count", title="Machines with Frequent Issues", color="Issue Count", color_continuous_scale="Reds")
            fig_mach.update_layout(height=350)
            st.plotly_chart(fig_mach, use_container_width=True)
        else:
            st.info("No machine-level failure data available for this period.")

# --- Tab 3: Reliability (FIXED DATE CRASH) ---
with tab_rel:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("##### Pareto: Top Downtime Causes")
        if not dt_f.empty:
            pareto = dt_f.groupby("cause_code")["duration_min"].sum().sort_values(ascending=False).reset_index()
            fig_par = px.bar(pareto, x="cause_code", y="duration_min", color="duration_min", color_continuous_scale="Reds")
            fig_par.update_layout(height=350, xaxis_title="Failure Code", yaxis_title="Minutes Lost")
            st.plotly_chart(fig_par, use_container_width=True)
            
    with col_r2:
        st.markdown("##### MTTR Analysis (Repair Speed)")
        if not dt_f.empty:
            # FIX: Ensure proper datetime conversion for trendline math
            dt_f['date_plot'] = pd.to_datetime(dt_f['start_time'].dt.date)
            mttr_trend = dt_f.groupby("date_plot")["duration_min"].mean().reset_index()
            
            fig_mttr = px.scatter(mttr_trend, x="date_plot", y="duration_min", 
                                  title="Avg Repair Time (Minutes)", 
                                  trendline="lowess")
            fig_mttr.update_layout(height=350)
            st.plotly_chart(fig_mttr, use_container_width=True)

# ======================================
# 10. AI Assistant (Native UI)
# ======================================
st.markdown("---")

col_header, col_clear = st.columns([6, 1])

with col_header:
    st.subheader("AI Factory Assistant")

with col_clear:
    # THE CLEAR BUTTON LOGIC
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I can analyze scrap trends, downtime logs, and standard procedures. What do you need?",
                "sources": []
            }
        ]
        st.rerun() # Forces the app to reload immediately

# Initialize Chat History (if not already done by the button)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I can analyze scrap trends, downtime logs, and standard procedures. What do you need?",
            "sources": []
        }
    ]

def clean_source_name(file_path):
    """Converts a full file path into a clean filename for display."""
    if not file_path:
        return "Unknown Source"
    return os.path.basename(file_path)

def contextualize_query(llm, history, latest_query):
    """
    Takes conversation history and the latest user question.
    If the user question references context (e.g., "it", "that machine"),
    rewrites it into a standalone question.
    """
    if not history:
        return latest_query
    
    # Keep only the last 2 exchanges to keep costs low and focus relevant
    recent_history = history[-4:] 
    
    history_str = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    system_prompt = f"""
    Given the chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just rewrite it if needed and otherwise return it as is.
    
    Chat History:
    {history_str}
    
    Latest User Question: {latest_query}
    
    Standalone Question:
    """
    
    # We use a separate lightweight call to get the rewritten query
    response = llm.invoke(system_prompt)
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I can analyze scrap trends, downtime logs, and standard procedures. What do you need?",
            "sources": [] # Key: Add empty sources list for init message
        }
    ]

# --- DISPLAY HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If this message has associated sources, show them in an expander
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Source Documents"):
                for source in msg["sources"]:
                    # Formatting the source display
                    source_name = clean_source_name(source.get("source", ""))
                    snippet = source.get("content", "")[:150].replace("\n", " ") + "..."
                    
                    st.markdown(f"**📄 {source_name}**")
                    st.caption(f"\"{snippet}\"")
                    st.divider()

# --- CHAT INPUT HANDLER ---
if prompt := st.chat_input("Ask about scrap, failures, or maintenance procedures..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if rag_error:
            response = f"⚠️ System Error: {rag_error}"
            message_placeholder.markdown(response)
        else:
            with st.spinner("Analyzing plant data..."):
                
                # --- STEP 1: CONTEXTUALIZE ---
                history_for_context = st.session_state.messages[:-1]
                standalone_query = contextualize_query(llm, history_for_context, prompt)
                
                # --- STEP 2: RETRIEVE ---
                docs = retriever.invoke(standalone_query)
                doc_text = "\n\n".join([d.page_content for d in docs])
                
                # --- STEP 3: ANSWER ---
                context_str = f"""
                Current Context:
                - Line: {sel_line} | Machine: {sel_machine}
                - OEE: {kpis['oee']}% | MTBF: {kpis['mtbf']}h
                - Financial Loss: ${kpis['financial_loss']:,}
                """
                
                final_prompt = f"""
                You are a factory assistant. Answer using the context below.
                
                DASHBOARD METRICS:
                {context_str}
                
                KNOWLEDGE BASE (SOPs/LOGS):
                {doc_text}
                
                USER QUESTION: {prompt}
                
                Answer concisely with bullet points if needed.
                """
                
                ai_response = llm.invoke(final_prompt)
                response = ai_response.content
                
                message_placeholder.markdown(response)
                
                # --- STEP 4: SHOW SOURCES IMMEDIATELY ---
                # Create a list of source dicts to save to history
                sources_data = []
                if docs:
                    with st.expander("📚 View Source Documents"):
                        for d in docs:
                            src_name = clean_source_name(d.metadata.get("source", ""))
                            # Save to list for history
                            sources_data.append({"source": src_name, "content": d.page_content})
                            
                            # Display now
                            st.markdown(f"**📄 {src_name}**")
                            st.caption(f"\"{d.page_content[:150].replace(chr(10), ' ')}...\"")
                            st.divider()

    # 3. Save Assistant Response WITH SOURCES to History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources_data # <--- This ensures they persist on refresh
    })

# ======================================
# Footer
# ======================================
st.markdown("---")
st.caption(f"All rights reserved  © {datetime.now().year}")
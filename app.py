import asyncio
import ast
import json
import math
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from selector_workflow import run_selector_customer_support

AGENTS = [
    ("Customer_Query_Agent", "1. Customer Query Agent"),
    ("Knowledge_Base_Agent", "2. Knowledge Base Agent"),
    ("Resolution_Agent", "3. Resolution Agent"),
    ("Log_Agent", "4. Log Agent"),
]


def parse_mapping(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except (ValueError, TypeError):
        try:
            data = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return None
    return data if isinstance(data, dict) else None


def normalized_confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def extract_agent_execution_steps(result):
    """Inspect actual agent messages and tool results, never infer success from order."""
    steps = {
        name: {"status": "Not Executed", "messages": [], "errors": [],
               "knowledge_result": None, "ticket_result": None}
        for name, _ in AGENTS
    }
    if isinstance(result, dict):
        return steps

    for message in getattr(result, "messages", []):
        name = getattr(message, "source", "")
        if name not in steps:
            continue
        kind = type(message).__name__
        content = getattr(message, "content", None)
        if kind == "ToolCallExecutionEvent":
            for item in content or []:
                tool_name = getattr(item, "name", "")
                raw = getattr(item, "content", "")
                if getattr(item, "is_error", False):
                    steps[name]["errors"].append(f"{tool_name}: {raw}")
                    continue
                if name == "Knowledge_Base_Agent" and tool_name == "search_company_knowledge":
                    parsed = parse_mapping(raw)
                    if (parsed is not None and isinstance(parsed.get("found"), bool)
                            and isinstance(parsed.get("sources"), list)
                            and isinstance(parsed.get("matches"), list)):
                        steps[name]["knowledge_result"] = parsed
                    else:
                        steps[name]["errors"].append("Invalid knowledge search result")
                if name == "Log_Agent" and tool_name == "log_customer_ticket":
                    parsed = parse_mapping(raw)
                    if parsed is not None and as_bool(parsed.get("logged")) and parsed.get("ticket_id"):
                        steps[name]["ticket_result"] = parsed
                    else:
                        steps[name]["errors"].append("Ticket logging was not confirmed")
        elif kind == "TextMessage" and isinstance(content, str) and content.strip():
            steps[name]["messages"].append(content)

    for name, details in steps.items():
        if details["errors"]:
            details["status"] = "Failed"
        elif name == "Customer_Query_Agent":
            allowed = {
                "category": {"Shipping", "Refund", "Account", "Product", "Billing", "Other"},
                "urgency": {"Low", "Medium", "High"},
                "sentiment": {"Positive", "Neutral", "Frustrated", "Angry"},
            }
            valid = any(
                (data := parse_mapping(msg)) is not None
                and all(data.get(key) in values for key, values in allowed.items())
                and isinstance(data.get("summary"), str) and bool(data["summary"].strip())
                for msg in details["messages"]
            )
            details["status"] = "Completed" if valid else (
                "Response Received — Unverified" if details["messages"] else "Not Executed"
            )
        elif name == "Knowledge_Base_Agent":
            kb = details["knowledge_result"]
            if kb is not None:
                details["status"] = "Completed — Policy Found" if kb["found"] else "Completed — No Relevant Policy"
            elif details["messages"]:
                details["status"] = "Response Received — Unverified"
        elif name == "Resolution_Agent":
            valid = any(
                (data := parse_mapping(msg)) is not None
                and isinstance(data.get("response"), str) and bool(data["response"].strip())
                and normalized_confidence(data.get("confidence")) is not None
                and isinstance(data.get("sources"), list)
                for msg in details["messages"]
            )
            details["status"] = "Completed" if valid else (
                "Response Received — Unverified" if details["messages"] else "Not Executed"
            )
        elif name == "Log_Agent":
            details["status"] = "Completed — Ticket Logged" if details["ticket_result"] else (
                "Response Received — Unverified" if details["messages"] else "Not Executed"
            )
    return steps


def extract_workflow_results(result):
    """Extract verified response and log data from workflow events."""
    output = {
        "response": "", "confidence": None, "sources": [], "ticket_id": "",
        "logged": False, "escalated": False, "escalation_reasons": [],
        "duplicate_request": False,
    }
    if isinstance(result, dict) and result.get("duplicate_request"):
        output.update({
            "duplicate_request": True,
            "ticket_id": str(result.get("ticket_id", "")),
            "logged": bool(result.get("ticket_id")),
            "response": "This request has already been processed. No new ticket was created.",
        })
        return output

    steps = extract_agent_execution_steps(result)
    for msg in steps["Resolution_Agent"]["messages"]:
        data = parse_mapping(msg)
        if (data and isinstance(data.get("response"), str)
                and data["response"].strip()):
            output["response"] = data["response"]
            output["confidence"] = normalized_confidence(data.get("confidence"))
            output["sources"] = data.get("sources") if isinstance(data.get("sources"), list) else []
    ticket = steps["Log_Agent"]["ticket_result"]
    if ticket:
        output["ticket_id"] = str(ticket["ticket_id"])
        output["logged"] = True
        output["escalated"] = as_bool(ticket.get("escalated"))
        reasons = ticket.get("escalation_reasons", [])
        output["escalation_reasons"] = reasons if isinstance(reasons, list) else ([str(reasons)] if reasons else [])
    return output


def style_dashboard_chart(chart):
    """Use high-contrast, semibold axis text across all analytics charts."""
    return (
        chart.configure_axis(
            labelColor="#17375D",
            titleColor="#12345A",
            labelFontSize=13,
            titleFontSize=15,
            labelFontWeight="bold",
            titleFontWeight="bold",
            labelPadding=9,
            titlePadding=13,
            gridColor="#E3EBF5",
            tickColor="#9AAEC6",
        )
        .configure_view(stroke=None)
    )


def badge(label, value, color):
    """Render readable badges using Streamlit's built-in status callouts."""
    text = f"**{label}:** {value}"
    if color == "red":
        st.error(text)
    elif color == "amber":
        st.warning(text)
    elif color == "green":
        st.success(text)
    else:
        st.info(text)


def render_execution_tracker(steps):
    st.markdown("### 🤖 Multi-Agent Execution Tracker")
    for name, title in AGENTS:
        details = steps[name]
        status = details["status"]
        icon = "🔴" if status == "Failed" else (
            "✅" if status.startswith("Completed") else (
                "🟡" if status.startswith("Response Received") else "⚪"
            )
        )
        with st.expander(f"{icon} {title} — {status}"):
            st.write(f"**Execution Status:** {status}")
            if name == "Customer_Query_Agent":
                for msg in details["messages"]:
                    data = parse_mapping(msg)
                    if data:
                        st.json(data)
                    else:
                        st.write(msg)
            elif name == "Knowledge_Base_Agent":
                kb = details["knowledge_result"]
                if kb is not None:
                    st.write("**Policy found:**", "Yes" if kb["found"] else "No")
                    st.write("**Sources:**", ", ".join(map(str, kb["sources"])) or "None")
                    for match in kb["matches"]:
                        if isinstance(match, dict):
                            st.write(f"**{match.get('source', 'Policy')} — {match.get('section', 'Section')}**")
                            st.write(match.get("content", ""))
                elif details["messages"]:
                    st.write(details["messages"][-1])
            elif name == "Log_Agent" and details["ticket_result"]:
                ticket = details["ticket_result"]
                st.write("**Ticket ID:**", ticket.get("ticket_id", ""))
                st.write("**Logged:**", "Yes")
                st.write("**Escalated:**", "Yes" if as_bool(ticket.get("escalated")) else "No")
                for reason in ticket.get("escalation_reasons", []) or []:
                    st.write(f"• {reason}")
            elif details["messages"]:
                for msg in details["messages"]:
                    data = parse_mapping(msg)
                    st.json(data) if data else st.write(msg)
            else:
                st.info("No final text response was captured.")
            for error in details["errors"]:
                st.error(str(error))
    st.divider()


def render_badges(steps, output):
    st.markdown("### 🏷️ Workflow Status Badges")
    classification = None
    for msg in steps["Customer_Query_Agent"]["messages"]:
        data = parse_mapping(msg)
        if data and steps["Customer_Query_Agent"]["status"] == "Completed":
            classification = data
            break
    urgency = classification.get("urgency") if classification else None
    kb = steps["Knowledge_Base_Agent"]["knowledge_result"]
    confidence = output["confidence"]
    cols = st.columns(3)
    with cols[0]:
        badge("Urgency", urgency or "Unknown", {"High": "red", "Medium": "amber", "Low": "green"}.get(urgency, "neutral"))
    with cols[1]:
        # UI-only bands; do not change existing escalation policy.
        if confidence is None:
            badge("Confidence", "Unavailable", "neutral")
        else:
            band = "High" if confidence >= 0.8 else "Moderate" if confidence >= 0.5 else "Low"
            badge("Confidence", f"{confidence:.0%} ({band})", "green" if band == "High" else "amber" if band == "Moderate" else "red")
    with cols[2]:
        badge("Knowledge", "Policy Found" if kb and kb["found"] else "No Relevant Policy" if kb else "Not Verified", "green" if kb and kb["found"] else "amber" if kb else "neutral")
    cols2 = st.columns(2)
    with cols2[0]:
        badge("Escalation", "Human Review Required" if output["escalated"] else "Not Required" if output["logged"] else "Unknown", "red" if output["escalated"] else "green" if output["logged"] else "neutral")
    with cols2[1]:
        badge("Ticket", "Logged" if output["logged"] else "Not Confirmed", "green" if output["logged"] else "red")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MyCSA - The AI Demo Store",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
:root { --csa-navy:#142D4E; --csa-blue:#2563EB; --csa-bg:#F3F7FC; --csa-line:#DCE5F0; }
[data-testid="stAppViewContainer"], .stApp { background: linear-gradient(155deg, #EFF5FD 0%, #F7FAFE 45%, #F3F7FC 100%); color:#172B4D; }
[data-testid="stHeader"] { background: rgba(243,247,252,.92); }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#142D4E 0%,#1B416A 100%); }
[data-testid="stSidebar"] * { color: #F3F8FF; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.17); }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius:10px; padding:8px 10px; transition:background .2s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.12); }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.20); }
.csa-hero { padding:32px 36px; border-radius:18px; background:linear-gradient(120deg,#102A43 0%,#1D4ED8 65%,#2563EB 100%); border-bottom:4px solid #38BDF8; box-shadow:0 10px 26px rgba(15,46,90,.17); margin-bottom:25px; position:relative; overflow:hidden; }
.csa-hero:after { content:""; position:absolute; right:-60px; top:-95px; width:270px; height:270px; border:1px solid rgba(255,255,255,.22); border-radius:50%; box-shadow:0 0 0 38px rgba(255,255,255,.045),0 0 0 76px rgba(255,255,255,.035); pointer-events:none; }
.csa-eyebrow { font-size:12px; font-weight:750; letter-spacing:.16em; color:#B9DAFF; margin-bottom:9px; }
.csa-hero h1.csa-title { font-size:clamp(27px,3vw,39px); line-height:1.25; font-weight:800; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF; letter-spacing:.2px; text-shadow:0 2px 8px rgba(0,0,0,.16); margin:0 0 10px; }
.csa-hero p.csa-subtitle { font-size:16px; color:#DBEAFE !important; -webkit-text-fill-color:#DBEAFE; margin:0; max-width:730px; line-height:1.6; }
.csa-brand { padding:8px 2px 4px; }
.csa-brand-name { font-size:27px; font-weight:800; color:#FFFFFF; }
.csa-brand-caption { font-size:13px; color:#B9DAFF; letter-spacing:.04em; }
[data-testid="stMain"] h2,[data-testid="stMain"] h3 { color:#17375D !important; -webkit-text-fill-color:#17375D; letter-spacing:-.02em; }
.csa-analytics-heading { color:#12345A !important; -webkit-text-fill-color:#12345A !important; font-size:1.75rem; font-weight:750; line-height:1.3; margin:0 0 1.1rem; }
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--csa-line); }
[data-testid="stMain"] [data-testid="stExpander"] { background:#FFFFFF; border:1px solid var(--csa-line); border-radius:13px; box-shadow:0 3px 12px rgba(20,45,78,.045); overflow:hidden; }
[data-testid="stMain"] [data-testid="stMetric"] { background:#FFFFFF; padding:17px 18px; border:1px solid var(--csa-line); border-radius:14px; box-shadow:0 3px 12px rgba(20,45,78,.05); }
[data-testid="stMain"] [data-testid="stMetricLabel"] { color:#52657E; }
[data-testid="stMain"] [data-testid="stMetricValue"] { color:#17375D; }
[data-testid="stMain"] textarea,[data-testid="stMain"] input { background:#FFFFFF; border-radius:10px; border-color:#C9D8EA; }
[data-testid="stMain"] textarea:focus,[data-testid="stMain"] input:focus { border-color:#2563EB; box-shadow:0 0 0 2px rgba(37,99,235,.12); }
[data-testid="stMain"] [data-testid="stButton"] button[kind="primary"] { background:#2563EB; color:#FFFFFF; border:1px solid #2563EB; border-radius:10px; font-weight:700; box-shadow:0 5px 12px rgba(37,99,235,.18); }
[data-testid="stMain"] [data-testid="stButton"] button[kind="primary"]:hover { background:#1D4ED8; border-color:#1D4ED8; }
[data-testid="stMain"] [data-testid="stButton"] button[kind="secondary"] { background:#FFFFFF; color:#1D4D83; border:1px solid #C8D9EE; border-radius:10px; font-weight:650; }
[data-testid="stMain"] [data-testid="stButton"] button[kind="secondary"]:hover { border-color:#2563EB; color:#1D4ED8; }
[data-testid="stMain"] [data-testid="stDataFrame"] { background:#FFFFFF; border:1px solid var(--csa-line); border-radius:12px; overflow:hidden; }
[data-testid="stMain"] [data-testid="stAlert"] { border-radius:11px; }
[data-testid="stMain"] hr { border-color:#DCE5F0; margin:22px 0; }
.csa-footer { color:#64748B; font-size:12px; text-align:center; padding:14px; }
@media (max-width:700px) { .csa-hero { padding:23px 20px; border-radius:15px; } [data-testid="stMainBlockContainer"] { padding-top:1rem; } }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="csa-brand"><div class="csa-brand-name">🤖 MyCSA</div>'
        '<div class="csa-brand-caption">THE AI DEMO STORE · SUPPORT</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Navigation")

    selected_page = st.radio(
        "Select Page",
        [
            "Customer Support",
            "Ticket History",
            "Dashboard Analytics"
        ],
        index=0,
        key="main_navigation"
    )

    st.divider()

    st.info(
        "Powered by Microsoft AutoGen "
        "and OpenAI."
    )

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<section class="csa-hero">'
    '<div class="csa-eyebrow">THE AI DEMO STORE · CUSTOMER EXPERIENCE</div>'
    '<h1 class="csa-title">🤖 MyCSA | AI Customer Support Assistant</h1>'
    '<p class="csa-subtitle">Intelligent Multi-Agent Customer Support Automation — transparent '
    'execution and policy-grounded resolutions in one workspace.</p>'
    '</section>',
    unsafe_allow_html=True,
)

if selected_page == "Customer Support":
    st.subheader("📝 Submit a Customer Complaint")
    complaint = st.text_area(
        "Describe your issue",
        placeholder="Example: My order delivery is delayed by three days and I have not received updates.",
        height=150,
        key="customer_complaint",
    )

    def clear_complaint():
        st.session_state.customer_complaint = ""

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        submit_button = st.button("🚀 Submit Complaint", use_container_width=True, type="primary")
    with button_col2:
        st.button("🔄 Clear / New Complaint", use_container_width=True, on_click=clear_complaint)

    st.divider()
    st.subheader("💬 AI Customer Support Response")
    if submit_button:
        if not complaint.strip():
            st.warning("Please enter your complaint before submitting.")
        else:
            try:
                with st.spinner("🤖 AI agents are processing your complaint..."):
                    result = asyncio.run(run_selector_customer_support(complaint.strip()))
                    workflow_output = extract_workflow_results(result)
                    agent_execution_steps = extract_agent_execution_steps(result)

                if workflow_output["duplicate_request"]:
                    st.info("This request was already processed. No new support ticket was created.")
                    st.metric("Existing Ticket ID", workflow_output["ticket_id"])
                else:
                    render_execution_tracker(agent_execution_steps)
                    render_badges(agent_execution_steps, workflow_output)
                    if workflow_output["response"]:
                        st.success("AI Response Generated Successfully")
                        st.markdown("### 💬 Customer Support Response")
                        st.write(workflow_output["response"])
                    else:
                        st.warning("The AI workflow did not return a valid response.")

                    if workflow_output["logged"]:
                        st.markdown("### 🎫 Support Ticket Details")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ticket ID", workflow_output["ticket_id"])
                        with col2:
                            conf = workflow_output["confidence"]
                            st.metric("Confidence", f"{conf:.0%}" if conf is not None else "Unavailable")
                        with col3:
                            st.metric("Escalation", "Yes" if workflow_output["escalated"] else "No")
                        if workflow_output["escalated"]:
                            st.warning("⚠️ This ticket requires human review.")
                            for reason in workflow_output["escalation_reasons"]:
                                st.write(f"• {reason}")
                        else:
                            st.success("✅ Your complaint has been processed without requiring escalation.")
                    else:
                        st.error("The support ticket could not be confirmed as saved. Please check the workflow logs.")

                    if workflow_output["sources"]:
                        with st.expander("📚 Knowledge Base Sources"):
                            for source in workflow_output["sources"]:
                                st.write(f"📄 {source}")
            except Exception as error:
                st.error("An error occurred while processing the customer complaint.")
                with st.expander("Technical Error Details"):
                    st.code(str(error))
    else:
        st.info("Your AI-generated response will appear here after submitting a complaint.")

# FOOTER
# --------------------------------------------------

st.divider()

# --------------------------------------------------
# TICKET HISTORY
# --------------------------------------------------

if selected_page in ("Ticket History", "Dashboard Analytics"):
    # Show the Ticket History title only on its own page.
    # The Dashboard Analytics page renders its dedicated heading below.
    if selected_page == "Ticket History":
        st.subheader("📋 Customer Support Ticket History")

    LOG_FILE = Path(__file__).resolve().parent / "logs" / "customer_tickets.csv"

    if LOG_FILE.exists():

        try:

            tickets_df = pd.read_csv(LOG_FILE)

            if not tickets_df.empty:

                if selected_page == "Dashboard Analytics":
                                # --------------------------------------------------
                    # DASHBOARD ANALYTICS - KPI METRICS
                    # --------------------------------------------------

                    st.markdown(
                        '<h2 class="csa-analytics-heading">📊 Customer Support Analytics</h2>',
                        unsafe_allow_html=True,
                    )

                    # Calculate total tickets
                    total_tickets = len(tickets_df)

                    # Identify escalated tickets
                    escalation_flags = (
                        tickets_df["escalated"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["true", "1"])
                    )

                    escalated_tickets = int(escalation_flags.sum())

                    # Calculate escalation rate
                    escalation_rate = (
                        escalated_tickets / total_tickets * 100
                        if total_tickets > 0
                        else 0
                    )

                    # Calculate average AI confidence
                    confidence_values = pd.to_numeric(
                        tickets_df["confidence"],
                        errors="coerce"
                    )

                    average_confidence = confidence_values.mean()

                    if pd.isna(average_confidence):
                        average_confidence = 0.0

                    # --------------------------------------------------
                    # DISPLAY KPI CARDS
                    # --------------------------------------------------

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                    with kpi1:
                        st.metric(
                            label="🎫 Total Tickets",
                            value=total_tickets
                        )

                    with kpi2:
                        st.metric(
                            label="⚠️ Escalated Tickets",
                            value=escalated_tickets
                        )

                    with kpi3:
                        st.metric(
                            label="📈 Escalation Rate",
                            value=f"{escalation_rate:.1f}%"
                        )

                    with kpi4:
                        st.metric(
                            label="🤖 Average AI Confidence",
                            value=f"{average_confidence:.1%}"
                        )

                    # --------------------------------------------------
                    # CUSTOMER COMPLAINT CATEGORY DISTRIBUTION
                    # --------------------------------------------------

                    st.divider()

                    st.markdown("### 📊 Complaints by Category")

                    # Count tickets in each category
                    category_counts = (
                        tickets_df["category"]
                        .fillna("Unknown")
                        .value_counts()
                        .rename_axis("Category")
                        .reset_index(name="Tickets")
                    )

                    # Horizontal bars keep category labels readable.
                    category_chart = (
                        alt.Chart(category_counts)
                        .mark_bar(color="#2563EB", cornerRadiusEnd=5)
                        .encode(
                            y=alt.Y("Category:N", sort="-x", title=None,
                                    axis=alt.Axis(labelAngle=0, labelColor="#17375D")),
                            x=alt.X("Tickets:Q", title="Tickets",
                                    axis=alt.Axis(tickMinStep=1)),
                            tooltip=["Category:N", "Tickets:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(style_dashboard_chart(category_chart), use_container_width=True)

                    # Display category summary
                    with st.expander("📋 View Category Breakdown"):
                        st.dataframe(
                            category_counts,
                            use_container_width=True,
                            hide_index=True
                        )

                    # --------------------------------------------------
                    # CUSTOMER COMPLAINT URGENCY DISTRIBUTION
                    # --------------------------------------------------

                    st.divider()

                    st.markdown("### 🚦 Complaints by Urgency")

                    # Count tickets by urgency
                    urgency_counts = (
                        tickets_df["urgency"]
                        .fillna("Unknown")
                        .value_counts()
                        .rename_axis("Urgency")
                        .reset_index(name="Tickets")
                    )

                    # Define the preferred urgency order
                    urgency_order = ["Low", "Medium", "High", "Unknown"]

                    # Sort urgency levels
                    urgency_counts["Urgency"] = pd.Categorical(
                        urgency_counts["Urgency"],
                        categories=urgency_order,
                        ordered=True
                    )

                    urgency_counts = (
                        urgency_counts
                        .sort_values("Urgency")
                        .reset_index(drop=True)
                    )

                    # Convert categories back to strings
                    urgency_counts["Urgency"] = (
                        urgency_counts["Urgency"].astype(str)
                    )

                    # Altair ensures readable, consistent axis titles and values.
                    urgency_chart = (
                        alt.Chart(urgency_counts)
                        .mark_bar(color="#2563EB", cornerRadiusEnd=5)
                        .encode(
                            x=alt.X("Urgency:N", sort=["Low", "Medium", "High", "Unknown"],
                                    axis=alt.Axis(labelAngle=0), title="Urgency"),
                            y=alt.Y("Tickets:Q", axis=alt.Axis(tickMinStep=1), title="Tickets"),
                            tooltip=["Urgency:N", "Tickets:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(style_dashboard_chart(urgency_chart), use_container_width=True)

                    # Display urgency summary table
                    with st.expander("📋 View Urgency Breakdown"):
                        st.dataframe(
                            urgency_counts,
                            use_container_width=True,
                            hide_index=True
                        )
                    # --------------------------------------------------
                    # DAILY CUSTOMER SUPPORT TICKET TRENDS
                    # --------------------------------------------------

                    st.divider()

                    st.markdown("### 📈 Daily Customer Support Ticket Trends")

                    # Create a copy to avoid modifying original data
                    trends_df = tickets_df.copy()

                    # Convert timestamp column to datetime
                    trends_df["timestamp"] = pd.to_datetime(
                        trends_df["timestamp"],
                        errors="coerce",
                        utc=True
                    )

                    # Remove records with invalid timestamps
                    trends_df = trends_df.dropna(subset=["timestamp"])

                    if not trends_df.empty:

                        # Convert timestamps to the store's reporting timezone
                        trends_df["ticket_date"] = (
                            trends_df["timestamp"]
                            .dt.tz_convert("Asia/Kolkata")
                            .dt.date
                        )

                        # Count tickets for each date
                        daily_counts = (
                            trends_df.groupby("ticket_date")
                            .size()
                            .reset_index(name="Tickets")
                        )

                        # Convert dates to datetime for plotting
                        daily_counts["ticket_date"] = pd.to_datetime(
                            daily_counts["ticket_date"]
                        )

                        # Sort chronologically
                        daily_counts = daily_counts.sort_values(
                            "ticket_date"
                        )

                        # Display daily ticket trend with stronger axis typography.
                        daily_chart = (
                            alt.Chart(daily_counts)
                            .mark_line(color="#2563EB", strokeWidth=3, point=True)
                            .encode(
                                x=alt.X("ticket_date:T", title="Date"),
                                y=alt.Y("Tickets:Q", title="Tickets",
                                        axis=alt.Axis(tickMinStep=1)),
                                tooltip=[alt.Tooltip("ticket_date:T", title="Date", format="%d %b %Y"),
                                         "Tickets:Q"],
                            )
                            .properties(height=320)
                        )
                        st.altair_chart(style_dashboard_chart(daily_chart), use_container_width=True)

                        # Display daily summary table
                        with st.expander("📋 View Daily Ticket Breakdown"):

                            st.dataframe(
                                daily_counts,
                                use_container_width=True,
                                hide_index=True
                            )

                    else:

                        st.info(
                            "No valid ticket timestamps are available "
                            "for trend analysis."
                        )

                if selected_page == "Ticket History":
                    # --------------------------------------------------
                    # TICKET SEARCH AND FILTERS
                    # --------------------------------------------------

                    # Display newest tickets first
                    tickets_df = tickets_df.iloc[::-1].reset_index(drop=True)

                    st.markdown("### 🔎 Search and Filter Tickets")

                    # Create three filter columns
                    filter_col1, filter_col2, filter_col3 = st.columns(3)

                    with filter_col1:

                        search_text = st.text_input(
                            "Search Ticket",
                            placeholder="Ticket ID or complaint",
                            key="ticket_search"
                        )

                    with filter_col2:

                        categories = sorted(
                            tickets_df["category"].dropna().unique().tolist()
                        )

                        selected_category = st.selectbox(
                            "Category",
                            ["All"] + categories,
                            key="ticket_category"
                        )

                    with filter_col3:

                        selected_escalation = st.selectbox(
                            "Escalation Status",
                            ["All", "Escalated", "Not Escalated"],
                            key="ticket_escalation"
                        )

                    # --------------------------------------------------
                    # APPLY FILTERS
                    # --------------------------------------------------

                    filtered_df = tickets_df.copy()

                    # Search by Ticket ID or complaint text
                    if search_text.strip():

                        search_mask = (
                            filtered_df["ticket_id"]
                            .fillna("")
                            .astype(str)
                            .str.contains(search_text.strip(), case=False, regex=False)
                            |
                            filtered_df["complaint"]
                            .fillna("")
                            .astype(str)
                            .str.contains(search_text.strip(), case=False, regex=False)
                        )

                        filtered_df = filtered_df[search_mask]

                    # Filter by category
                    if selected_category != "All":

                        filtered_df = filtered_df[
                            filtered_df["category"] == selected_category
                        ]

                    # Filter by escalation status
                    if selected_escalation != "All":

                        escalation_values = (
                            filtered_df["escalated"]
                            .astype(str)
                            .str.lower()
                            .isin(["true", "1"])
                        )

                        if selected_escalation == "Escalated":

                            filtered_df = filtered_df[escalation_values]

                        else:

                            filtered_df = filtered_df[~escalation_values]

                    # --------------------------------------------------
                    # DISPLAY FILTERED RESULTS
                    # --------------------------------------------------

                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------
                    # DOWNLOAD FILTERED TICKET HISTORY
                    # --------------------------------------------------

                    st.markdown("### 📥 Export Ticket History")

                    # Convert filtered tickets to CSV format
                    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="📥 Download Filtered Tickets (CSV)",
                        data=csv_data,
                        file_name="MyCSA_Filtered_Tickets.csv",
                        mime="text/csv",
                        disabled=filtered_df.empty,
                        use_container_width=True,
                        key="download_filtered_tickets"
                    )
                else:

                    st.info("No customer support tickets found.")

        except Exception as error:

            st.error(
                f"Unable to load ticket history: {error}"
            )

    else:

        st.info(
            "Ticket history will appear after the first "
            "customer complaint is processed."
        )

    st.caption(
        "MyCSA | Intelligent Multi-Agent Customer Support Automation "
        "| The AI Demo Store"
    )

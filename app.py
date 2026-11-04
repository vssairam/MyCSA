import asyncio
import ast
import json

from pathlib import Path
import pandas as pd

import streamlit as st

from selector_workflow import run_selector_customer_support

def extract_workflow_results(result):
    """
    Extract the resolution and ticket information
    from the AutoGen multi-agent workflow.
    """

    # --------------------------------------------------
    # SESSION STATE INITIALIZATION
    # --------------------------------------------------

    if "customer_complaint" not in st.session_state:
        st.session_state.customer_complaint = ""

    output = {
        "response": "",
        "confidence": 0.0,
        "sources": [],
        "ticket_id": "",
        "logged": False,
        "escalated": False,
        "escalation_reasons": [],
        "duplicate_request": False
    }

    # Handle duplicate requests returned by SelectorGroupChat
    if isinstance(result, dict) and result.get("duplicate_request"):

        output["duplicate_request"] = True

        output["ticket_id"] = result.get("ticket_id", "")
        output["logged"] = True

        output["response"] = (
            "This request has already been processed. "
            "No new support ticket was created. "
            f"Existing Ticket ID: {output['ticket_id']}"
        )

        return output

    for message in result.messages:

        # Extract the Resolution Agent's JSON response
        if message.source == "Resolution_Agent":

            if type(message).__name__ == "TextMessage":

                try:
                    data = json.loads(message.content)

                    if isinstance(data, dict) and "response" in data:
                        output["response"] = data["response"]
                        output["confidence"] = data.get(
                            "confidence", 0.0
                        )
                        output["sources"] = data.get(
                            "sources", []
                        )

                except (json.JSONDecodeError, TypeError):
                    pass

        # Extract the Log Agent's tool execution result
        if (
            message.source == "Log_Agent"
            and type(message).__name__ == "ToolCallExecutionEvent"
        ):

            for tool_result in message.content:

                if (
                    tool_result.name == "log_customer_ticket"
                    and not tool_result.is_error
                ):

                    try:
                        ticket_data = ast.literal_eval(
                            tool_result.content
                        )

                        if ticket_data.get("logged") is True:

                            output["ticket_id"] = ticket_data.get(
                                "ticket_id", ""
                            )

                            output["logged"] = True

                            output["escalated"] = ticket_data.get(
                                "escalated", False
                            )

                            output["escalation_reasons"] = ticket_data.get(
                                "escalation_reasons", []
                            )

                    except (ValueError, SyntaxError, TypeError):
                        pass

    return output                                                                                                                                                                                                                                                                                           

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

.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #1E3A8A;
}

.sub-title {
    font-size: 18px;
    color: #64748B;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 MyCSA")

    st.markdown("### The AI Demo Store")

    st.divider()

    st.markdown("### Navigation")

    st.markdown("🏠 Customer Support")
    st.markdown("📋 Ticket History")
    st.markdown("📊 Dashboard Analytics")

    st.divider()

    st.info(
        "Powered by Microsoft AutoGen "
        "and OpenAI."
    )

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Customer Support Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Welcome to The AI Demo Store Customer Support Portal'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# CUSTOMER COMPLAINT FORM
# --------------------------------------------------

st.subheader("📝 Submit a Customer Complaint")

complaint = st.text_area(
    "Describe your issue",
    placeholder=(
        "Example: My order delivery is delayed "
        "by three days and I have not received updates."
    ),
    height=150,
    key="customer_complaint"
)

# --------------------------------------------------
# COMPLAINT ACTION BUTTONS
# --------------------------------------------------

def clear_complaint():
    st.session_state.customer_complaint = ""


button_col1, button_col2 = st.columns(2)

with button_col1:
    submit_button = st.button(
        "🚀 Submit Complaint",
        use_container_width=True,
        type="primary"
    )

with button_col2:
    st.button(
        "🔄 Clear / New Complaint",
        use_container_width=True,
        on_click=clear_complaint
    )

# --------------------------------------------------
# RESPONSE SECTION
# --------------------------------------------------

st.divider()

st.subheader("💬 AI Customer Support Response")

if submit_button:

    if not complaint.strip():

        st.warning(
            "Please enter your complaint before submitting."
        )

    else:

        try:

            # Display progress while agents are working
            with st.spinner(
                "🤖 AI agents are processing your complaint..."
            ):

                # Execute the four-agent workflow
                result = asyncio.run(
                    run_selector_customer_support(complaint.strip())
                )

                # Extract the workflow results
                workflow_output = extract_workflow_results(result)

            # ------------------------------------------
            # DISPLAY CUSTOMER RESOLUTION
            # ------------------------------------------

            if workflow_output["response"]:

                st.success("AI Response Generated Successfully")

                st.markdown("### 💬 Customer Support Response")

                st.write(workflow_output["response"])

            else:

                st.warning(
                    "The AI workflow did not return a valid response."
                )

            # ------------------------------------------
            # DISPLAY TICKET INFORMATION
            # ------------------------------------------

            if workflow_output["logged"]:

                if workflow_output["duplicate_request"]:
                    st.info(
                        "This request was already processed. "
                        "No new support ticket was created."
                    )

                    st.metric(
                        "Existing Ticket ID",
                        workflow_output["ticket_id"]
                    )
                else:
                    st.markdown("### 🎫 Support Ticket Details")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Ticket ID",
                            workflow_output["ticket_id"]
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f'{workflow_output["confidence"]:.0%}'
                        )

                    with col3:
                        st.metric(
                            "Escalation",
                            "Yes" if workflow_output["escalated"]
                            else "No"
                        )

                    # Display escalation warning
                    if workflow_output["escalated"]:

                        st.warning(
                            "⚠️ This ticket requires human review."
                        )

                        for reason in workflow_output[
                            "escalation_reasons"
                        ]:
                            st.write(f"• {reason}")

                    else:

                        st.success(
                            "✅ Your complaint has been processed "
                            "without requiring escalation."
                        )

            else:

                st.error(
                    "The support ticket could not be confirmed "
                    "as saved. Please check the workflow logs."
                )

            # ------------------------------------------
            # DISPLAY KNOWLEDGE BASE SOURCES
            # ------------------------------------------

            if workflow_output["sources"]:

                with st.expander(
                    "📚 Knowledge Base Sources"
                ):

                    for source in workflow_output["sources"]:
                        st.write(f"📄 {source}")

        except Exception as error:

            st.error(
                "An error occurred while processing "
                "the customer complaint."
            )

            # Show technical details for debugging
            with st.expander("Technical Error Details"):
                st.code(str(error))

else:

    st.info(
        "Your AI-generated response will appear here "
        "after submitting a complaint."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

# --------------------------------------------------
# TICKET HISTORY
# --------------------------------------------------

st.divider()

st.subheader("📋 Customer Support Ticket History")

LOG_FILE = Path(__file__).resolve().parent / "logs" / "customer_tickets.csv"

if LOG_FILE.exists():

    try:

        tickets_df = pd.read_csv(LOG_FILE)

        if not tickets_df.empty:

                        # --------------------------------------------------
            # DASHBOARD ANALYTICS - KPI METRICS
            # --------------------------------------------------

            st.divider()

            st.subheader("📊 Customer Support Analytics")

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

            # Display category distribution chart
            st.bar_chart(
                category_counts,
                x="Category",
                y="Tickets",
                use_container_width=True
            )

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

            # Display urgency distribution chart
            st.bar_chart(
                urgency_counts,
                x="Urgency",
                y="Tickets",
                use_container_width=True
            )

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

                # Display daily ticket trend
                st.line_chart(
                    daily_counts,
                    x="ticket_date",
                    y="Tickets",
                    use_container_width=True
                )

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
    "MyCSA | Multi-Agent Customer Support Automation "
    "| The AI Demo Store"
)
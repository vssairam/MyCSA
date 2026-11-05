# MyCSA v1.2 – Multi-Agent Customer Support Automation

## 1. Project Overview

**MyCSA (Multi-Agent Customer Support Automation)** is an AI-powered customer support application developed for a fictional e-commerce company called **The AI Demo Store**.

The application uses four specialized AI agents to classify customer complaints, retrieve relevant company policies, generate policy-grounded responses, and automatically create customer support tickets.

MyCSA combines **Microsoft AutoGen, SelectorGroupChat, Retrieval-Augmented Generation (RAG), FAISS, OpenAI models, and Streamlit** to deliver an interactive multi-agent customer support experience.

Version 1.2 introduces improvements in workflow reliability, knowledge-base grounding, escalation handling, ticket logging integrity, and the Streamlit user interface.

### Key Features

* **Multi-Agent Automation:** Four specialized AutoGen agents collaborate to process customer complaints.
* **Intelligent Complaint Classification:** Identifies complaint category, urgency, sentiment, and summary.
* **FAISS-Based Knowledge Retrieval:** Retrieves relevant information from company policy documents.
* **Policy-Grounded Responses:** Generates customer responses using retrieved knowledge-base information.
* **Source Traceability:** Displays relevant knowledge-base sources to support response verification.
* **Automated Ticket Creation:** Generates unique customer support ticket IDs.
* **Intelligent Escalation:** Identifies complaints requiring human review based on confidence, knowledge availability, and predefined escalation rules.
* **Duplicate Logging Prevention:** Validates workflow completion and prevents duplicate ticket creation during supported retry scenarios.
* **Request Status Tracking:** Maintains PENDING, PROCESSING, COMPLETED, and FAILED states.
* **Idempotency Handling:** Identifies previously processed request IDs to avoid unnecessary reprocessing.
* **CSV Audit Logging:** Records ticket details, confidence scores, sources, escalation decisions, and escalation reasons.
* **Three-Page Streamlit Interface:**

  * Customer Support
  * Ticket History
  * Dashboard Analytics
* **Ticket Search and Export:** Supports searching, filtering, and downloading ticket records.
* **Dashboard Analytics:** Displays ticket metrics, complaint categories, urgency distribution, and daily trends.
* **Automated Testing:** Includes component tests, workflow regression tests, escalation validation, and CSV integrity checks.

### Version 1.2 Enhancements

| Enhancement Area         | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| Workflow Reliability     | Improved workflow termination and successful ticket logging validation          |
| Knowledge Base Grounding | Improved policy-based responses and source traceability                         |
| Escalation Logic         | Improved confidence validation and handling of sensitive complaints             |
| CSV Logging              | Validated ticket IDs, timestamps, escalation consistency, and logging integrity |
| Streamlit Navigation     | Introduced three independent application pages                                  |
| Dashboard Analytics      | Preserved KPI metrics, category distribution, urgency charts, and daily trends  |
| Testing                  | Added regression coverage and pytest configuration for asynchronous tests       |

MyCSA v1.2 is intended as a functional demonstration of multi-agent AI automation rather than a production-hardened customer support platform.

---

## 2. Project Objectives

The primary objective of MyCSA is to demonstrate how Generative AI and multi-agent systems can automate customer support operations.

### Specific Objectives

**Objective 1: Automate Complaint Classification**

Identify the complaint category, urgency, sentiment, and summary using a specialized Query Agent.

**Objective 2: Retrieve Relevant Company Policies**

Use a Knowledge Base Agent and FAISS vector database to retrieve policy information relevant to the customer's complaint.

**Objective 3: Generate Policy-Grounded Responses**

Use a Resolution Agent to generate customer support responses based on retrieved company policies rather than unsupported assumptions.

**Objective 4: Automate Ticket Creation**

Use a Log Agent to create customer support tickets and record relevant complaint information.

**Objective 5: Support Escalation**

Identify requests requiring human review according to predefined escalation rules.

**Objective 6: Track Request Processing**

Maintain request statuses:

* PENDING
* PROCESSING
* COMPLETED
* FAILED

**Objective 7: Reduce Duplicate Processing**

Use request IDs and an idempotency registry to identify requests that have already been processed.

**Objective 8: Provide an Interactive Dashboard**

Enable users to submit complaints, view AI-generated responses, review support tickets, and monitor ticket history through Streamlit.

---

## 3. Technology Stack

| Technology        | Purpose                                       |
| ----------------- | --------------------------------------------- |
| Python            | Core application development                  |
| Microsoft AutoGen | Multi-agent orchestration                     |
| SelectorGroupChat | Agent selection and workflow coordination     |
| OpenAI API        | Large language model inference                |
| LangChain         | Document processing and embedding integration |
| FAISS             | Vector database for semantic policy search    |
| Streamlit         | Interactive web application                   |
| Pandas            | Ticket history and analytics                  |
| JSON              | Request status and idempotency registries     |
| CSV               | Customer support ticket storage               |
| python-dotenv     | Environment variable configuration            |
| VS Code           | Development environment                       |

### Architecture Summary

MyCSA uses four specialized agents:

**1. Customer Query Agent**

Classifies customer complaints and identifies important information.

**2. Knowledge Base Agent**

Retrieves relevant company policies using semantic search.

**3. Resolution Agent**

Generates customer support responses grounded in retrieved policy information.

**4. Log Agent**

Creates support tickets and determines whether human escalation is required.

The agents are coordinated using **AutoGen SelectorGroupChat**, which follows the configured processing sequence.

### Application Workflow

Customer Complaint → Query Agent → Knowledge Base Agent → FAISS → Resolution Agent → Log Agent → Ticket Creation → Request Status Update → Streamlit Dashboard

---
## 4. Project Folder Structure

The MyCSA application is organized into separate modules for AI agents, knowledge retrieval, workflow orchestration, ticket management, request tracking, and the Streamlit interface.

```text
MyCSA/
│
├── agents/
│   ├── query_agent.py
│   ├── knowledge_agent.py
│   ├── resolution_agent.py
│   └── log_agent.py
│
├── knowledge_base/
│   ├── shipping_policy.md
│   ├── refund_policy.md
│   ├── account_and_login.md
│   ├── product_faq.md
│   └── billing_policy.md
│
├── logs/
│   ├── customer_tickets.csv
│   ├── idempotency_registry.json
│   └── request_status_registry.json
│
├── vector_store/
│   ├── policies.index
│   └── metadata.json
│
├── app.py
├── selector_workflow.py
├── multi_agent_workflow.py
│
├── build_vector_store.py
├── semantic_search.py
├── kb_search.py
│
├── idempotency_utils.py
├── idempotency_registry.py
├── request_status.py
├── request_status_registry.py
│
├── test_autogen.py
├── test_csv_integrity.py
├── test_escalation_rules.py
├── test_duplicate_logging.py
├── test_exactly_one_logging.py
├── test_idempotency_registry.py
├── test_knowledge_agent.py
├── test_log_agent.py
├── test_multi_agent_scenarios.py
├── test_query_agent.py
├── test_request_status_registry.py
├── test_resolution_agent.py
├── test_selector_termination.py
├── test_semantic_search.py
│
├── .env
├── .env.example
├── .gitignore
├── .gitattributes
├── pytest.ini
├── requirements.txt
├── README.md
│
├── venv/
└── __pycache__/
```

*Note: The nested files shown under `agents/`, `knowledge_base/`, `logs/`, and `vector_store/` reflect the project components developed and used during implementation. The top-level files were verified against the project directory listing.*

### 4.1 Core Application Files

| File                         | Responsibility                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `app.py`                     | Streamlit interface for complaint submission, AI responses, ticket details, and dashboard |
| `selector_workflow.py`       | Main SelectorGroupChat workflow coordinating the four AI agents                           |
| `multi_agent_workflow.py`    | Original RoundRobinGroupChat implementation retained for reference                        |
| `build_vector_store.py`      | Builds the FAISS vector database from company policy documents                            |
| `semantic_search.py`         | Performs semantic searches against the FAISS vector database                              |
| `kb_search.py`               | Knowledge-base search functionality                                                       |
| `idempotency_utils.py`       | Generates unique request identifiers                                                      |
| `idempotency_registry.py`    | Records processed request IDs and their associated ticket IDs                             |
| `request_status.py`          | Defines request processing statuses                                                       |
| `request_status_registry.py` | Stores and retrieves request processing status information                                |

### 4.2 AI Agent Modules

| Agent                | File                         | Responsibility                                                |
| -------------------- | ---------------------------- | ------------------------------------------------------------- |
| Query Agent          | `agents/query_agent.py`      | Classifies complaints and extracts relevant information       |
| Knowledge Base Agent | `agents/knowledge_agent.py`  | Retrieves relevant company policy information                 |
| Resolution Agent     | `agents/resolution_agent.py` | Generates policy-grounded customer responses                  |
| Log Agent            | `agents/log_agent.py`        | Creates customer support tickets and applies escalation rules |

### 4.3 Data Storage

MyCSA uses three types of local storage:

**FAISS Vector Database**

Stores searchable representations of company policy documents.

**CSV Ticket Storage**

Stores customer support ticket records in `logs/customer_tickets.csv`.

**JSON Registries**

Store request processing information:

* `logs/idempotency_registry.json` maps processed request IDs to ticket IDs.
* `logs/request_status_registry.json` records request statuses and associated ticket IDs.

### 4.4 Testing Modules

The project contains Python test programs for validating agent behavior, semantic search, workflow execution, duplicate logging prevention, idempotency, request status tracking, and termination conditions.

### 4.5 Development and Configuration Files

* `.env` — Stores local environment configuration and API credentials. This file must not be committed to source control.
* `.env.example` — Provides an example environment configuration.
* `.gitignore` — Specifies files and directories excluded from Git tracking.

## 5. Multi-Agent Architecture

MyCSA uses four specialized AI agents coordinated through Microsoft AutoGen's `SelectorGroupChat`.

Each agent has a defined responsibility, allowing the application to separate complaint classification, knowledge retrieval, response generation, and ticket creation.

### 5.1 Architecture Diagram

```text
                  CUSTOMER
                     |
                     v
               STREAMLIT UI
                  (app.py)
                     |
                     v
           SELECTOR GROUP CHAT
            (selector_workflow.py)
                     |
                     v
              +-------------+
              | QUERY AGENT |
              +-------------+
                     |
                     v
           +------------------+
           | KNOWLEDGE AGENT  |
           +------------------+
                     |
                     v
             FAISS VECTOR DB
                     |
                     v
           +------------------+
           | RESOLUTION AGENT |
           +------------------+
                     |
                     v
              +-----------+
              | LOG AGENT |
              +-----------+
                     |
                     v
             SUPPORT TICKET
                     |
                     v
           IDEMPOTENCY REGISTRY
                     |
                     v
             REQUEST STATUS
                     |
                     v
             STREAMLIT RESULTS
```

### 5.2 Query Agent

**Module:** `agents/query_agent.py`

The Query Agent analyzes the customer's complaint and extracts structured information.

Its responsibilities include:

* Identifying the complaint category.
* Determining urgency.
* Analyzing customer sentiment.
* Generating a concise complaint summary.

The structured output helps subsequent agents understand the customer's issue.

### 5.3 Knowledge Base Agent

**Module:** `agents/knowledge_agent.py`

The Knowledge Base Agent retrieves relevant company policies.

It uses the `search_company_knowledge` tool to search the FAISS vector database.

The knowledge base contains policy documents covering shipping, refunds, billing, account access, and product-related questions.

The retrieved information provides supporting context for the Resolution Agent.

### 5.4 Resolution Agent

**Module:** `agents/resolution_agent.py`

The Resolution Agent generates a customer-facing response based on the complaint and retrieved company policies.

Its responsibilities include:

* Producing a clear customer support response.
* Using retrieved policies to support the resolution.
* Providing a confidence score.
* Identifying relevant knowledge sources.
* Flagging cases requiring human review.

### 5.5 Log Agent

**Module:** `agents/log_agent.py`

#### Intelligent Ticket Escalation — MyCSA v1.2

MyCSA v1.2 includes enhanced escalation handling to identify customer complaints that may require human intervention.

The Log Agent evaluates structured information received from the preceding agents and applies predefined escalation rules.

**Escalation considerations include:**

* **Confidence Validation:** Confidence values are checked for validity and expected range. Invalid or non-finite values are handled conservatively.
* **Knowledge Base Availability:** Complaints without sufficient supporting knowledge-base information may require human review.
* **Human Review Flags:** Explicit human-review indicators are normalized and considered during escalation.
* **Sensitive Complaint Detection:** The escalation logic recognizes complaint patterns involving duplicate charges, double payments, and other configured sensitive issues.
* **Escalation Reasons:** When escalation is triggered, the corresponding reasons are recorded for review.

The implementation also normalizes boolean inputs to reduce inconsistent escalation decisions caused by different representations of true and false values.

These improvements help make escalation decisions more consistent, explainable, and auditable.

#### CSV Ticket Audit Logging — MyCSA v1.2

MyCSA stores customer support ticket information in:

`logs/customer_tickets.csv`

Each ticket contains structured information that supports operational monitoring and audit review.

**Ticket fields:**

| Field                | Description                                      |
| -------------------- | ------------------------------------------------ |
| `ticket_id`          | Unique customer support ticket identifier        |
| `timestamp`          | Ticket creation date and time                    |
| `complaint`          | Original customer complaint                      |
| `category`           | Classified complaint category                    |
| `urgency`            | Identified urgency level                         |
| `sentiment`          | Customer sentiment                               |
| `summary`            | Concise complaint summary                        |
| `resolution`         | Generated customer support response              |
| `confidence`         | Resolution confidence score                      |
| `sources`            | Knowledge-base references                        |
| `kb_found`           | Indicates whether relevant knowledge was found   |
| `escalated`          | Indicates whether the ticket requires escalation |
| `escalation_reasons` | Reasons supporting the escalation decision       |

**Data Integrity Validation**

The CSV integrity validation checks include:

* Presence of required ticket fields.
* Unique ticket identifiers.
* Valid ticket ID formatting.
* Valid timestamps.
* Consistency between escalation status and escalation reasons.
* Detection of missing or duplicate records.

**Validation Results**

During Phase 4, the CSV integrity checks were executed against 79 existing ticket records.

| Validation Metric          | Result |
| -------------------------- | ------ |
| Total records reviewed     | 79     |
| Unique ticket IDs          | 79     |
| Duplicate ticket IDs       | 0      |
| Missing required values    | 0      |
| Escalation inconsistencies | 0      |
| Escalated tickets          | 29     |
| Non-escalated tickets      | 50     |

These results represent the dataset validated during Phase 4. The number of records may change as additional customer complaints are processed.

**Data Privacy**

Ticket logs and local request registries are operational data files and should not be committed to a public repository when they contain customer information.

The application should generate or maintain these files locally during execution.

---

### 5.6 SelectorGroupChat Orchestration

**Module:** `selector_workflow.py`

MyCSA uses `SelectorGroupChat` to coordinate the four agents.

The workflow follows this processing sequence:

1. Query Agent
2. Knowledge Base Agent
3. Resolution Agent
4. Log Agent

A custom selector function, `select_next_agent()`, determines which agent should execute next.

The workflow also uses a termination condition to identify successful ticket logging.

### 5.7 Workflow Completion Validation

MyCSA validates successful ticket creation before treating a request as completed.

The workflow checks that:

* A successful ticket logging event exists.
* Exactly one successful ticket logging event occurred.
* A valid ticket ID was returned.

The workflow then registers the request ID and ticket ID in the idempotency registry and updates the request status to `COMPLETED`.

If an exception occurs during the protected workflow execution, the request status is updated to `FAILED`.

### 5.8 Request Processing Lifecycle

The application uses the following request statuses:

| Status     | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| PENDING    | Request has been registered for processing                  |
| PROCESSING | Multi-agent workflow is executing                           |
| COMPLETED  | Ticket creation has been validated and processing completed |
| FAILED     | An error occurred during workflow execution                 |

### 5.9 Idempotency Handling

Each new request receives a unique request ID in the format `REQ-XXXXXXXXXXXX`.

Before starting the multi-agent workflow, the application checks whether that request ID has already been processed.

## 6. Installation, Configuration, and Execution

### 6.1 Prerequisites

Before running MyCSA, ensure the following are available:

* Windows operating system
* Python and pip
* Visual Studio Code
* Internet connectivity
* An OpenAI API key
* Access to the MyCSA project files

### 6.2 Open the Project

Open the MyCSA project folder in Visual Studio Code.

Launch the integrated PowerShell terminal.

All commands below should be executed from the project's root directory.

### 6.3 Create a Python Virtual Environment

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

When activation succeeds, the terminal prompt should begin with `(venv)`.

### 6.4 Install Dependencies

Install the required Python packages:

```powershell
python -m pip install -r requirements.txt
```

The `requirements.txt` file was generated from the working development environment using `pip freeze`.

It may contain additional development packages beyond the minimum application requirements.

### 6.5 Configure Environment Variables

The application uses environment variables for its OpenAI configuration.

Create a local `.env` file using `.env.example` as a reference.

Configure the following values:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Replace the placeholder with your own valid API key.

**Security:** Never publish your actual API key or commit the `.env` file to Git.

### 6.6 Build the FAISS Vector Database

MyCSA uses FAISS to search company policy documents.

The source documents are stored in the `knowledge_base/` directory.

To build the vector database, execute:

```powershell
python build_vector_store.py
```

The build process creates the searchable FAISS index and associated metadata.

An OpenAI API key and internet connectivity are required when generating embeddings through the OpenAI API.

### 6.7 Run the Streamlit Application

Start the application using:

```powershell
streamlit run app.py
```

Streamlit will display a local application URL in the terminal.

Open that URL in your browser to access the MyCSA customer support interface.

### 6.8 Submit a Customer Complaint

In the Streamlit interface:

1. Enter a customer complaint.
2. Click **Submit Complaint**.
3. Wait for the multi-agent workflow to finish.
4. Review the AI-generated customer support response.
5. Check the generated support ticket ID.
6. Review the confidence score and escalation status.
7. Expand the knowledge-base sources section when sources are available.
8. Review the ticket history and dashboard.

### 6.9 Run the SelectorGroupChat Workflow Directly

### Streamlit User Interface — MyCSA v1.2

MyCSA v1.2 provides an interactive Streamlit application with three independent navigation pages accessible through the sidebar.

#### 1. Customer Support

The Customer Support page provides the primary interface for submitting customer complaints.

**Features:**

* Enter and submit customer complaints.
* Execute the four-agent AutoGen workflow.
* Display AI-generated customer support responses.
* Display unique support ticket IDs.
* Show confidence scores and escalation status.
* Display escalation reasons when applicable.
* Provide knowledge-base source references for verification.

**Workflow:**

Customer Complaint → Query Agent → Knowledge Base Agent → Resolution Agent → Log Agent → Customer Response and Ticket Creation

#### 2. Ticket History

The Ticket History page provides access to previously processed customer support tickets stored in the local CSV file.

**Features:**

* Display existing customer support ticket records.
* Search tickets by ticket ID or complaint.
* Filter tickets using the available filtering controls.
* Review ticket details and escalation information.
* Export filtered ticket records to CSV.

**Data Source:** `logs/customer_tickets.csv`

#### 3. Dashboard Analytics

The Dashboard Analytics page provides an overview of customer support operations using information recorded in the ticket history.

**Features:**

* Total ticket count.
* Ticket escalation metrics.
* Complaint category distribution.
* Urgency distribution.
* Daily ticket volume trends.
* Daily ticket breakdown.

The dashboard uses Pandas for data processing and Streamlit for visualization.

#### Navigation

Users can switch between the three pages using the sidebar navigation menu.

| Navigation Option   | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| Customer Support    | Submit complaints and review AI-generated resolutions |
| Ticket History      | Search, filter, review, and export ticket records     |
| Dashboard Analytics | Monitor customer support metrics and trends           |

Each page is displayed independently, improving usability and separating complaint processing from historical reporting and analytics.

---

### 6.10 Verify Request Status

To inspect saved request statuses, execute:

```powershell
python -c "from request_status_registry import load_request_statuses; import json; print(json.dumps(load_request_statuses(), indent=4))"
```

Successfully completed requests should appear with a `COMPLETED` status and an associated ticket ID.

### 6.11 Run Existing Tests

MyCSA includes test programs for its major components.

For example:

```powershell
python test_semantic_search.py
```

```powershell
python test_knowledge_agent.py
```

```powershell
python test_idempotency_registry.py
```

```powershell
python test_request_status_registry.py
```

The tests help validate knowledge retrieval, idempotency, and request status tracking.

### 6.12 Stop the Application

To stop the Streamlit server, return to the terminal running Streamlit and press:

**Ctrl + C**

---

## 7. Testing and Validation

MyCSA v1.2 was validated through component testing, multi-agent workflow regression testing, knowledge-base grounding checks, escalation validation, CSV integrity checks, and end-to-end Streamlit testing.

### 7.1 Component Testing

The following tests were completed during development.

| Component               | Validation                | Result                      |
| ----------------------- | ------------------------- | --------------------------- |
| FAISS Semantic Search   | Knowledge retrieval tests | PASS — 11/11                |
| Knowledge Base Agent    | Agent retrieval tests     | PASS — 6/6                  |
| Multi-Agent Workflow    | Scenario validation       | PASS — 5/5                  |
| Idempotency Registry    | Registry functionality    | PASS — 5/5                  |
| Request Status Registry | Status management         | PASS — 6/6                  |
| Escalation Logic        | Escalation unit tests     | PASS — 13/13                |
| CSV Integrity           | Ticket data validation    | PASS — 79 records, 0 issues |

These results reflect successful test executions during development. They are not a guarantee that every possible input or operating condition has been tested.

### 7.2 Workflow Reliability Validation

Phase 1 focused on improving the reliability of the four-agent workflow.

The following areas were validated:

* SelectorGroupChat termination behavior.
* Exactly-one successful ticket logging.
* Duplicate ticket prevention.
* Workflow completion validation.
* Multi-agent integration scenarios.

| Validation                   | Result     |
| ---------------------------- | ---------- |
| Selector termination         | PASS — 5/5 |
| Exactly-one logging          | PASS — 4/4 |
| Duplicate logging prevention | PASS — 6/6 |
| Multi-agent integration      | PASS — 5/5 |

### 7.3 Knowledge Base Grounding and Source Traceability

Phase 2 focused on ensuring that generated customer responses were supported by retrieved company policies.

Five integration scenarios were validated:

1. Shipping-related complaint.
2. Refund-related complaint.
3. Duplicate payment complaint.
4. Account login issue.
5. Unrelated flight-related question.

The tests evaluated:

* Retrieval of relevant company policies.
* Appropriate source references.
* Policy-grounded responses.
* Avoidance of unsupported policy claims.
* Appropriate handling of questions outside the available knowledge base.

**Result: PASS — 5/5 integration scenarios.**

### 7.4 Escalation Logic Validation

Phase 3 introduced improvements to ticket escalation handling.

The validation covered:

* Boolean normalization for knowledge availability and human-review indicators.
* Confidence-score validation.
* Handling of non-finite confidence values.
* Sensitive complaint detection.
* Duplicate payment and double-charge complaint patterns.
* Escalation reason consistency.

**Result: PASS — 13/13 escalation unit tests.**

A subsequent multi-agent regression run also passed its five scenarios.

### 7.5 CSV Audit and Data Integrity Validation

Phase 4 validated the customer support ticket CSV.

The validation covered:

* Required fields.
* Unique ticket identifiers.
* Ticket ID formatting.
* Timezone-aware timestamps.
* Escalation status consistency.
* Missing and duplicate records.

| Metric                     | Result |
| -------------------------- | ------ |
| Records validated          | 79     |
| Unique ticket IDs          | 79     |
| Duplicate ticket IDs       | 0      |
| Missing required values    | 0      |
| Escalation inconsistencies | 0      |
| Escalated tickets          | 29     |
| Non-escalated tickets      | 50     |

**Result: PASS — 79 records validated with no detected integrity issues.**

These figures represent the Phase 4 validation snapshot and may change as new tickets are created.

### 7.6 Streamlit UI Validation

Phase 5 introduced three independent navigation pages.

| Feature                        | Result |
| ------------------------------ | ------ |
| Customer Support navigation    | PASS   |
| Ticket History navigation      | PASS   |
| Dashboard Analytics navigation | PASS   |
| Customer complaint submission  | PASS   |
| AI-generated response display  | PASS   |
| Ticket ID display              | PASS   |
| Escalation status display      | PASS   |
| Knowledge-base source display  | PASS   |
| Ticket history updates         | PASS   |
| Dashboard analytics updates    | PASS   |
| Ticket search and filtering    | PASS   |
| CSV export                     | PASS   |

The final end-to-end Streamlit test confirmed that a shipping-related complaint could be submitted, processed, logged, and reflected in the dashboard.

### 7.7 Automated Testing with pytest

MyCSA v1.2 includes pytest and pytest-asyncio for executing automated tests.

The project contains a `pytest.ini` configuration file to support asynchronous test execution.

**Run the multi-agent integration test:**

```powershell
python -m pytest -q test_multi_agent_scenarios.py
```

**Latest verified result:**

```text
1 passed in 49.91s
```

The pytest result represents one asynchronous test function. The function may execute multiple internal scenarios.

**Run the escalation tests:**

```powershell
python test_escalation_rules.py

**Final MyCSA v1.2 Regression Validation**

| Validation | Final Result |
|---|---|
| Escalation unit tests | PASS — 13/13 |
| CSV integrity validation | PASS — 90 tickets, 0 issues |
| Multi-agent workflow regression | PASS — 1 pytest test in 55.50 seconds |

The escalation and CSV integrity checks are standalone Python validation
scripts. The multi-agent regression test is executed using pytest.

All three final regression checks passed successfully.
```

**Run the CSV integrity validation:**

```powershell
python test_csv_integrity.py
```

**Run the complete pytest-discovered test suite:**

```powershell
python -m pytest -q
```

The complete suite command is provided for reproducibility; a successful full-suite result should only be reported after it has been executed and verified.

### 7.8 Request Status and Idempotency Validation

The application maintains the following request processing statuses:

* PENDING
* PROCESSING
* COMPLETED
* FAILED

The workflow validates successful ticket creation before recording a request as completed.

Sequential retries using an already-processed request ID return the previously associated ticket rather than creating another ticket.

These behaviors were validated during development.

### 7.9 Testing Scope and Limitations

MyCSA v1.2 is intended as a functional multi-agent customer support demonstration.

The completed tests do not establish:

* Concurrent duplicate-request protection.
* Distributed or multi-server processing.
* High-volume production load handling.
* Production-grade security and monitoring.
* Comprehensive recovery testing for every possible failure condition.

The application should undergo additional security, performance, concurrency, and operational testing before being considered for production deployment.

---


## 8. Demonstration Scenarios

MyCSA can be demonstrated using realistic customer complaints from The AI Demo Store.

The following scenarios showcase the application's multi-agent processing, policy retrieval, response generation, ticket logging, and escalation capabilities.

### 8.1 Scenario 1 – Delayed Delivery

**Customer Complaint:**

> My order was supposed to arrive three days ago, but I have not received it. I have also not received any delivery updates.

**Expected Demonstration:**

1. The customer submits the complaint through Streamlit.
2. The Query Agent identifies the delivery-related issue.
3. The Knowledge Base Agent retrieves relevant shipping policies.
4. The Resolution Agent generates a policy-grounded response.
5. The Log Agent creates a customer support ticket.
6. The application displays the response and ticket details.
7. The request status is recorded as COMPLETED after successful processing.

**Validation:** Confirm that the response, ticket ID, and relevant policy sources appear in Streamlit.

### 8.2 Scenario 2 – Refund Request

**Customer Complaint:**

> I returned a product last week, but I have not received my refund. Please let me know when I can expect it.

**Expected Demonstration:**

The Knowledge Base Agent should retrieve relevant refund policy information.

The Resolution Agent should generate a response based on the available policy.

The Log Agent should create a support ticket.

**Validation:** Review the generated response, policy sources, and ticket details.

### 8.3 Scenario 3 – Account Access Issue

**Customer Complaint:**

> I am unable to log in to my account even after resetting my password. Please help me access my account.

**Expected Demonstration:**

The Query Agent should classify the complaint as an account-related issue.

The Knowledge Base Agent should retrieve relevant account and login policies.

The Resolution Agent should provide appropriate guidance based on the retrieved information.

**Validation:** Confirm that the response uses relevant account support information.

### 8.4 Scenario 4 – Billing Issue

**Customer Complaint:**

> I was charged twice for the same order. Please investigate the duplicate payment and help me get a refund.

**Expected Demonstration:**

The Query Agent should identify the billing-related complaint.

The Knowledge Base Agent should retrieve relevant billing policies.

The Resolution Agent should provide a policy-grounded response.

The Log Agent should create a ticket and apply the configured escalation rules.

**Validation:** Check the ticket details and whether the application flags the request for human review.

### 8.5 Scenario 5 – Idempotency Verification

**Objective:** Demonstrate that an already-processed request ID does not trigger another workflow execution during a sequential retry.

**Demonstration Method:**

Use the Python workflow interface to submit a request with a specific request ID.

After successful completion, submit the same request ID again.

**Expected Result:**

* The workflow identifies the request ID as previously processed.
* The existing ticket ID is returned.
* The four-agent workflow is not executed again.
* No additional support ticket is created by the retry.

**Note:** The standard Streamlit submission flow generates a new request ID for each submission. Re-entering the same complaint in Streamlit is therefore not equivalent to retrying the same request ID.

### 8.6 Scenario 6 – Request Status Verification

After processing a complaint, execute:

```powershell
python -c "from request_status_registry import load_request_statuses; import json; print(json.dumps(load_request_statuses(), indent=4))"
```

**Expected Result:**

The request status registry displays the processed request ID, its COMPLETED status, and the corresponding ticket ID.

### 8.7 Final Demonstration Checklist

| Verification Item             | Expected Result                                                       |
| ----------------------------- | --------------------------------------------------------------------- |
| Streamlit application starts  | Application opens successfully                                        |
| Customer complaint submission | Complaint is accepted                                                 |
| Query Agent                   | Complaint is classified                                               |
| Knowledge Base Agent          | Relevant policy information is retrieved                              |
| Resolution Agent              | Customer response is generated                                        |
| Log Agent                     | Support ticket is created                                             |
| Ticket ID                     | Displayed in Streamlit                                                |
| Confidence score              | Displayed in Streamlit                                                |
| Escalation status             | Displayed in Streamlit                                                |
| Knowledge sources             | Available when retrieved                                              |
| Ticket history                | Existing records remain accessible                                    |
| Dashboard                     | Displays ticket information                                           |
| Request status                | COMPLETED after successful processing                                 |
| Idempotency                   | Sequential retry with the same request ID returns the existing ticket |

**Important:** The scenarios describe expected behavior. Actual responses, confidence scores, and escalation decisions depend on the complaint, retrieved policies, and configured agent rules.

---
## 9. Architecture and Workflow Diagrams

MyCSA includes two Mermaid diagrams documenting its architecture and customer complaint processing workflow.

### 9.1 System Architecture Diagram

The System Architecture Diagram illustrates the interaction between the following components:

* Streamlit user interface
* AutoGen SelectorGroupChat
* Query Agent
* Knowledge Base Agent
* Resolution Agent
* Log Agent
* FAISS vector database
* Customer support ticket storage
* Idempotency registry
* Request status registry
* Streamlit dashboard

**View Diagram:** [MyCSA System Architecture](diagrams/system_architecture.md)

### 9.2 Customer Complaint Processing Workflow

The Customer Support Workflow Diagram illustrates the complete complaint processing lifecycle.

It includes request ID generation, idempotency verification, agent execution, ticket logging, request status transitions, and error handling.

**View Diagram:** [MyCSA Customer Support Workflow](diagrams/customer_support_workflow.md)

### 9.3 Viewing the Diagrams

Both diagrams are written using Mermaid syntax inside Markdown files.

To view them in Visual Studio Code:

1. Open the relevant diagram file.
2. Press **Ctrl + Shift + V** to open Markdown Preview.
3. Review the rendered diagram.

These diagrams are intended to support project documentation, technical reviews, and demonstrations.

---

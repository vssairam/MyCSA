# MyCSA – Multi-Agent Customer Support Automation

## 1. Project Overview

**MyCSA (Multi-Agent Customer Support Automation)** is an AI-powered customer support application developed for a fictional e-commerce company called **The AI Demo Store**.

The application uses multiple specialized AI agents to process customer complaints, retrieve relevant company policies, generate appropriate responses, and create customer support tickets.

MyCSA combines **Microsoft AutoGen, SelectorGroupChat, Retrieval-Augmented Generation (RAG), FAISS, OpenAI models, and Streamlit** to provide an automated customer support experience.

The application also includes request status tracking, idempotency checks, ticket escalation, and a customer support dashboard.

### Key Features

* Automated customer complaint classification.
* Multi-agent collaboration using AutoGen SelectorGroupChat.
* Semantic search across company policies using FAISS.
* Policy-grounded customer support responses.
* Automated customer support ticket creation.
* Ticket escalation based on predefined rules.
* Request status tracking.
* Idempotency checks to identify previously processed request IDs.
* Customer support dashboard using Streamlit.
* Ticket history and analytics.

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

The Log Agent records the processed complaint using the `log_customer_ticket` tool.

Its responsibilities include:

* Creating a unique support ticket ID.
* Saving ticket information to `logs/customer_tickets.csv`.
* Applying predefined escalation rules.
* Returning ticket creation results.

Ticket IDs use the format `CSA-XXXXXXXX`.

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

The SelectorGroupChat workflow can also be executed from the terminal:

```powershell
python selector_workflow.py
```

This is useful for demonstrating agent execution and inspecting workflow output without using Streamlit.

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

MyCSA was validated through component-level tests, multi-agent workflow execution, and end-to-end Streamlit testing.

### 7.1 Component Testing

The following components were tested during development.

| Component               | Validation                   | Result       |
| ----------------------- | ---------------------------- | ------------ |
| FAISS Semantic Search   | Knowledge retrieval tests    | PASS – 11/11 |
| Knowledge Base Agent    | Agent retrieval tests        | PASS – 6/6   |
| Multi-Agent Workflow    | Workflow scenario tests      | PASS – 5/5   |
| Idempotency Registry    | Registry functionality tests | PASS – 5/5   |
| Request Status Registry | Status management tests      | PASS – 6/6   |

These results reflect the successful test executions recorded during development.

### 7.2 SelectorGroupChat Workflow Validation

The SelectorGroupChat workflow was tested by submitting a customer complaint through `selector_workflow.py`.

The execution confirmed:

* Successful request ID generation.
* Request status transitions through PENDING, PROCESSING, and COMPLETED.
* Execution of all four agents.
* Retrieval of relevant company policies.
* Generation of a customer support response.
* Successful creation of a support ticket.
* Exactly one successful ticket logging event.
* Registration of the request ID and ticket ID in the idempotency registry.

### 7.3 Streamlit Integration Validation

The integrated Streamlit application was tested successfully.

The following features were verified:

| Feature                                | Result |
| -------------------------------------- | ------ |
| Customer complaint submission          | PASS   |
| AI-generated customer support response | PASS   |
| Support ticket ID display              | PASS   |
| Existing ticket history                | PASS   |
| Dashboard functionality                | PASS   |
| Request status persistence             | PASS   |

### 7.4 Verified Request Status Records

The request status registry contained the following successfully completed requests:

| Request ID       | Ticket ID    | Status    |
| ---------------- | ------------ | --------- |
| REQ-BCFBA604261E | CSA-66A1C79D | COMPLETED |
| REQ-044A819CA9B8 | CSA-DEBEE789 | COMPLETED |

These records demonstrate that the application successfully persisted request statuses and associated ticket IDs.

### 7.5 Idempotency Validation

The workflow checks whether a request ID has already been processed.

When an existing request ID is submitted again, the workflow returns the associated ticket ID without executing the multi-agent workflow again.

This behavior was validated for sequential retries during development.

### 7.6 Testing Scope and Limitations

The successful tests demonstrate that the implemented application functions correctly for the tested scenarios.

However, the following capabilities have not been established by these results:

* Concurrent duplicate-request protection.
* Distributed or multi-server processing.
* High-volume load handling.
* Production-grade security and monitoring.
* Comprehensive recovery testing for every possible failure condition.

MyCSA Version 1.0 is intended as a functional multi-agent customer support demonstration rather than a production-hardened service.

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

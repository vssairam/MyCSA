# MyCSA – System Architecture

The following diagram illustrates the major components of the MyCSA customer support automation application.

```mermaid
flowchart TD
    Customer["Customer"] --> UI["Streamlit UI<br/>app.py"]

    UI --> Workflow["AutoGen SelectorGroupChat<br/>selector_workflow.py"]

    Workflow --> Query["Query Agent"]
    Query --> Knowledge["Knowledge Base Agent"]
    Knowledge --> Resolution["Resolution Agent"]
    Resolution --> Log["Log Agent"]

    Knowledge --> Search["Semantic Search"]
    Search --> FAISS[("FAISS Vector Database")]
    FAISS --> Search
    Search --> Knowledge

    Log --> Tickets[("Customer Tickets<br/>CSV")]

    Workflow --> Idempotency[("Idempotency Registry<br/>JSON")]
    Workflow --> Status[("Request Status Registry<br/>JSON")]

    Workflow --> UI

    UI --> Dashboard["Ticket History and Dashboard"]
    Tickets --> Dashboard

    classDef interface fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef agent fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef storage fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef workflow fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class UI,Dashboard interface
    class Query,Knowledge,Resolution,Log agent
    class FAISS,Tickets,Idempotency,Status storage
    class Workflow workflow
```

## Architecture Description

The customer submits a complaint through the Streamlit interface.

The SelectorGroupChat workflow coordinates four specialized agents:

1. The Query Agent classifies the complaint.
2. The Knowledge Base Agent retrieves relevant company policies using FAISS.
3. The Resolution Agent generates a policy-grounded response.
4. The Log Agent creates a support ticket and applies escalation rules.

The workflow also maintains request status information and an idempotency registry.

After processing, Streamlit displays the customer support response, ticket details, and available knowledge sources.

The dashboard reads stored ticket information to display ticket history and analytics.

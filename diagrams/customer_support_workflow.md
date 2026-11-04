# MyCSA – Customer Complaint Processing Workflow

This diagram illustrates how MyCSA processes customer complaints using AutoGen SelectorGroupChat, FAISS, and request tracking.

## Workflow Diagram

```mermaid
flowchart TD
    A([Customer Submits Complaint]) --> B[Generate or Receive Request ID]

    B --> C{Request ID Already Processed?}

    C -->|Yes| D[Retrieve Existing Ticket ID]
    D --> E([Return Existing Ticket Without Reprocessing])

    C -->|No| F[Set Status: PENDING]
    F --> G[Set Status: PROCESSING]

    G --> H[Query Agent: Classify Complaint]
    H --> I[Knowledge Base Agent: Retrieve Policies]
    I --> J[FAISS Semantic Search]
    J --> K[Resolution Agent: Generate Response]
    K --> L[Log Agent: Create Support Ticket]

    L --> M{Exactly One Successful Ticket Log?}

    M -->|Yes| N[Register Request ID and Ticket ID]
    N --> O[Set Status: COMPLETED]
    O --> P[Return Workflow Results]
    P --> Q([Display Response and Ticket in Streamlit])

    M -->|No| R[Raise Workflow Error]
    R --> S[Set Status: FAILED]

    G -.->|Workflow Exception| S
    H -.->|Exception| S
    I -.->|Exception| S
    J -.->|Exception| S
    K -.->|Exception| S
    L -.->|Exception| S
    N -.->|Exception| S
    O -.->|Exception| S

    S --> T([Report Processing Error])

    classDef startEnd fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef agent fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef status fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef decision fill:#f3e8ff,stroke:#9333ea,color:#581c87
    classDef error fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    class A,E,Q,T startEnd
    class H,I,J,K,L agent
    class F,G,O,S status
    class C,M decision
    class R error
```

## Workflow Explanation

### 1. Request Initialization

When a customer submits a complaint, MyCSA generates a unique request ID.

The workflow can also accept an existing request ID supplied programmatically.

### 2. Idempotency Verification

The workflow checks whether the request ID is already registered.

If the request ID has already been processed, the existing ticket ID is returned without running the agents again.

### 3. Request Status Tracking

New requests move through the following statuses:

**PENDING → PROCESSING → COMPLETED**

If an exception occurs during the protected workflow execution, the status changes to **FAILED**.

### 4. Multi-Agent Processing

The SelectorGroupChat coordinates four agents in sequence:

1. **Query Agent:** Classifies the complaint.
2. **Knowledge Base Agent:** Retrieves relevant company policies using FAISS.
3. **Resolution Agent:** Generates a policy-grounded customer response.
4. **Log Agent:** Creates the support ticket and applies escalation rules.

### 5. Ticket Validation

After the agents finish, MyCSA checks whether exactly one successful ticket logging event occurred.

A successful result allows the request ID and ticket ID to be registered.

### 6. Workflow Completion

The request status is updated to COMPLETED.

The workflow results are returned to Streamlit, which displays the response and ticket information.

### 7. Error Handling

Exceptions during the protected workflow execution are recorded with a FAILED status and propagated to the calling application.

**Note:** This diagram represents the main processing flow. Initialization failures occurring before the workflow's exception handler are outside the illustrated FAILED-status handling path.

---



```mermaid
graph TD
    A[DataFrame] --> B[cynde.functional.embed]
    A --> C[cynde.functional.generate]
    A --> D[cynde.functional.predict.train]
    A --> E[cynde.functional.predict.predict]

    B --> F{Embedding Method}
    F --> G[OpenAI API]
    F --> H[Deploy TEI Server]
    F --> I[JSON Caching]
    G --> J[list[float]]
    H --> J
    I --> J
    J --> K[DataFrame]

    C --> L{Generation Method}
    L --> M[OpenAI API]
    L --> N[Deploy TGI Server]
    L --> O[JSON Caching]
    M --> P{Output Type}
    N --> P
    O --> P
    P --> Q[str]
    P --> R[enum]
    P --> S[struct]
    Q --> T[DataFrame]
    R --> T
    S --> T

    D --> U[Train Boosted Trees]
    U --> V[Deploy Model Server]

    E --> W[Predict with Boosted Trees]
    W --> X[Predictions]
    X --> Y[DataFrame]

    Z[Pydantic Model] --> C
```
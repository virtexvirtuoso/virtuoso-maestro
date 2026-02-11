# Maestro UX Redesign - User Flow

## Current vs Proposed Flow

### Current Flow (Problematic)
```mermaid
flowchart TD
    A[Load Home Page] --> B[Show Chart with hardcoded XBTUSD]
    A --> C[Show Empty Form]
    C --> D[User waits for dropdowns to load]
    D --> E[User fills all 10+ fields at once]
    E --> F[Click Run]
    F --> G[Hope it works]
```

### Proposed Flow (Improved)
```mermaid
flowchart TD
    A[Load Home Page] --> B{Has Data Sources?}
    B -->|No| C[Empty State: Guide to Data Page]
    B -->|Yes| D[Show Data Source Summary Cards]

    D --> E[User Clicks 'New Optimization']
    E --> F[Step 1: Select Data Source]
    F --> G[Load symbols for provider]
    G --> H[Show date range available]
    H --> I[Step 2: Configure Strategy]
    I --> J[Load strategy params dynamically]
    J --> K[Step 3: Set Test Parameters]
    K --> L[Review Summary]
    L --> M[Run Optimization]
    M --> N[Real-time Progress]
    N --> O[Results with guided analysis]
```

## Screen State Machine

```mermaid
stateDiagram-v2
    [*] --> Loading: Page Load
    Loading --> EmptyState: No data sources
    Loading --> ReadyState: Has data

    EmptyState --> DataManagement: Click 'Add Data'
    DataManagement --> ReadyState: Data downloaded

    ReadyState --> WizardStep1: Click 'New Test'
    WizardStep1 --> WizardStep2: Provider+Symbol selected
    WizardStep2 --> WizardStep3: Strategy configured
    WizardStep3 --> ReviewStep: Parameters set
    ReviewStep --> Running: Confirm run

    Running --> Completed: Success
    Running --> Failed: Error

    Completed --> Evaluation: View results
    Failed --> WizardStep1: Retry
```

## Navigation Architecture

```mermaid
flowchart LR
    subgraph Primary["Primary Navigation"]
        Home[Home/Dashboard]
        Results[Results List]
        Evaluate[Evaluation]
        Compare[Compare]
        Data[Data Sources]
    end

    subgraph Secondary["Contextual Actions"]
        NewTest[+ New Test]
        QuickRun[Quick Run]
        Export[Export]
    end

    Home --> NewTest
    Home --> Results
    Results --> Evaluate
    Evaluate --> Compare
    Data --> Home
```

## Component Dependencies

```mermaid
flowchart TD
    subgraph DataLayer["Data Dependencies"]
        Provider[Provider Selection]
        Symbol[Symbol Selection]
        DateRange[Date Range]
        Strategy[Strategy Selection]
        Params[Strategy Parameters]
    end

    Provider -->|triggers| Symbol
    Provider -->|triggers| DateRange
    Symbol -->|validates| DateRange
    Strategy -->|loads| Params

    subgraph FormState["Form Validation"]
        Valid{All Required?}
        Submit[Enable Submit]
    end

    DateRange --> Valid
    Params --> Valid
    Valid -->|Yes| Submit
```

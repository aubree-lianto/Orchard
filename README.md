# Orchard

Orchard is a production-style, API-based LLM research agent designed to accelerate technical research workflows. It uses LangGraph for deterministic agent orchestration and deploys GPU-backed LLM inference on Kubernetes, following modern MLOps best practices.

This project emphasizes scalability, reliability, and infrastructure-aware system design rather than prompt-centric experimentation.

---

## Motivation

Research workflows frequently involve repetitive and time-consuming steps such as:

- Searching for relevant sources
- Reading and filtering documents
- Synthesizing findings
- Producing structured summaries

While large language models are well-suited for these tasks, deploying them in a reliable, scalable, and observable way remains challenging. Orchard explores how agent-based LLM systems can be deployed as real backend services using production infrastructure patterns.

---

## What This Project Does

Orchard exposes a simple API for structured research tasks.

### API Example

```http
POST /research
```

Given a research query, the system:

1. Plans a research strategy
2. Executes tool-based information gathering
3. Synthesizes findings using an LLM
4. Returns structured, source-aware results

The focus of the project is backend architecture and MLOps design, not user interface development.

---

### High-Level Architecture

```
Client
  |
Ingress
  |
FastAPI (LangGraph Agent Service)
  |
  |-- Tool Calls (search, fetch, summarize)
  |
  |-- Vector Store / Cache
  |
  |-- LLM Inference Service (vLLM, GPU)
```

---

### Architectural Principles

- Separation of concerns between API logic and model inference

- Stateless services for horizontal scalability

- Independent scaling of CPU-bound and GPU-bound workloads

## Core Components

### Agent Orchestration

Orchard uses LangChain and LangGraph to model research as a deterministic, stateful workflow.

- LangChain provides tool abstractions

- LangGraph defines the research process as a state machine

This approach improves debuggability, observability, and reliability compared to single-prompt pipelines.

### LLM Inference

- Open-source LLMs served using vLLM

- GPU-backed inference pods deployed on Kubernetes

- Support for LoRA adapters for task-specific fine-tuning

Inference is deployed as a separate service to allow independent scaling and updates.

### Infrastructure

- Docker for containerization
- Kubernetes for deployment and orchestration
- Horizontal Pod Autoscaling for API services
- NVIDIA device plugin for GPU scheduling (when available)

### ### Training and Model Lifecycle

- LoRA fine-tuning jobs are executed on a Slurm-based compute cluster
- Trained adapters are versioned and stored separately from base models
- Adapters can be loaded dynamically at inference time

This separation mirrors real-world ML infrastructure, where training and serving have distinct requirements.

## Repository Structure

orchard/
├── api/              # FastAPI service and LangGraph agents
├── inference/        # GPU-backed LLM inference service
├── training/         # LoRA fine-tuning jobs
├── deploy/           # Kubernetes manifests and Helm charts
├── observability/    # Metrics and monitoring configuration
├── ci/               # CI/CD pipelines
└── docs/             # Architecture and design documentation

---

## Scalability Design

Orchard is designed to scale across multiple dimensions:

- Request volume: API services scale horizontally

- Model load: inference services scale independently

- Model evolution: LoRA adapters enable rapid iteration without full redeployments

This design avoids monolithic services and supports incremental system growth.

---

## Observability

Observability is treated as a first-class concern.

The system includes:

- Request latency and throughput metrics

- LLM usage tracking (token counts and errors)

- Health checks for API and inference services

These metrics enable effective monitoring and debugging in production environments.

---

## Why This Project Exists

This project is intentionally infrastructure-heavy.

Orchard is designed to demonstrate:

- Production-style LLM deployment

- Kubernetes-native ML systems

- MLOps lifecycle thinking

- Agent orchestration beyond prompt chaining

Rather than optimizing for novelty, Orchard prioritizes clarity, correctness, and scalability.

---

## Future Work

- Multi-tenant research routing
- Streaming responses

- Improved retrieval and ranking

- Automated retraining pipelines

- Model evaluation and benchmarking
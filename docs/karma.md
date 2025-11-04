# KARMA Multi-Agent Knowledge Graph Enrichment

This document describes the KARMA-style multi-agent enrichment pipeline that is now built into `llmserver-rs`. The implementation reuses the local RKLLM-backed model instances exposed by the server and orchestrates them through a set of deterministic prompts inspired by the "KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment" paper.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    KARMA Orchestrator                    │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────┐ │
│  │ Planner      │    │ Extractor     │    │ Validator  │ │
│  │ Agent        │    │ Agent         │    │ Agent      │ │
│  │ (strategy)   │    │ (triple gen.) │    │ (QA & QC)  │ │
│  └──────────────┘    └───────────────┘    └────────────┘ │
│           ▲                    ▲                  ▲      │
│           │                    │                  │      │
│           └───────────── shared RKLLM pool ───────┘      │
└──────────────────────────────────────────────────────────┘
```

1. **Planner agent** summarises the operator goal, current graph and instructions, and returns a structured plan (objective, tasks, success metrics).
2. **Extractor agent** analyses each submitted document in parallel (serial execution for deterministic ordering) to propose candidate triples.
3. **Validator agent** verifies each candidate against the supporting document and the current graph state before it is merged.
4. The orchestrator merges accepted triples, records rejected ones, and maintains an audit log containing prompts, raw model responses and parsing status per agent call.

All agents are orchestrated by the in-tree [`ajeto`](../src/ajeto/mod.rs) engine. The engine mirrors the execution primitives of the upstream [porkbrain/ajeto](https://github.com/porkbrain/ajeto) project and binds them directly to the RKLLM-backed actor pool that already powers chat completions. Each agent invocation is routed through the shared `ProcessMessages` recipients, which keeps the implementation hot-swappable with existing deployments and avoids any additional worker processes.

## API Specification

The REST endpoint is exposed under `/v1/knowledge/karma/enrich`.

### Request Body

```
POST /v1/knowledge/karma/enrich
Content-Type: application/json
Authorization: Bearer <token>

{
  "model": "kautism/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4",
  "graph": {
    "nodes": [
      {"id": "node-1", "label": "Example Entity", "properties": {"type": "Concept"}}
    ],
    "edges": []
  },
  "documents": [
    "Document text describing a new relation between Example Entity and Sample Entity."
  ],
  "goal": "Augment the supply chain knowledge graph with logistics providers",
  "instructions": "Prefer high-confidence supplier relationships only"
}
```

### Response Body

```
{
  "plan": {
    "objective": "…",
    "tasks": ["…"],
    "success_metrics": ["…"]
  },
  "updated_graph": {
    "nodes": [ … ],
    "edges": [ … ]
  },
  "new_nodes": [ … ],
  "new_edges": [ … ],
  "accepted_candidates": [
    {
      "subject": "Example Entity",
      "predicate": "partner_of",
      "object": "Sample Entity",
      "confidence": 0.86,
      "justification": "…"
    }
  ],
  "rejected_candidates": [
    {
      "candidate": { "subject": "…" },
      "reason": "Validator agent returned an unparsable response"
    }
  ],
  "agent_logs": [
    {
      "agent": "extractor",
      "prompt": "…",
      "response": "…",
      "timestamp_ms": 1739999999999,
      "parsed_successfully": true
    }
  ]
}
```

## Operational Notes

- **Model reuse** – The orchestrator reuses the already loaded RKLLM actors. It does not spin up additional model instances, ensuring that the feature can be deployed without extra GPU/NPUs.
- **Timeouts** – Each agent invocation is limited by a configurable timeout (30 seconds by default). Calls that exceed the timeout are reported back as validation errors without aborting the entire enrichment run.
- **Result determinism** – Candidate limits (default 8 per document) and sequential validation are used to keep output predictable even when multiple LLM workers are available.
- **Hot swapping** – Because the orchestration layer is pure Rust and leverages the existing actor pool, it can be updated or replaced without restarting the RKLLM runtime.

## Extensibility

Future improvements can introduce:

- Specialised agents for entity linking or ontology alignment.
- External tooling hooks (search, retrieval) by extending the validator prompts.
- Persistence adapters that automatically push the enriched graph to a backing store.
- Reusable agent templates contributed upstream to the `ajeto` project, allowing additional workflows (e.g. ontology alignment or QA) to reuse the same execution substrate.


# LLM Server Control Plane

This document describes the new secure management plane that powers the unified text and speech serving stack.

## Overview

The binary now boots with three core services:

1. **Model Manager** – dynamically starts and stops LLM, ASR, and TTS actors at runtime.
2. **Credential Vault** – stores API keys, Hugging Face access tokens, and usage metrics inside an encrypted SQLite database.
3. **Admin Web Console** – password protected UI for provisioning models, API keys, and downloads.

All user-facing inference endpoints require API key authentication. Token usage is approximated from prompt and completion characters and persisted for cost accounting.

## Startup Requirements

Set a 32-byte base64 master key before launching:

```bash
export LLMSERVER_MASTER_KEY="$(openssl rand -base64 32)"
```

Optional environment variables:

- `LLMSERVER_DATABASE` – SQLite path (default `data/llm-admin.db`).
- `LLMSERVER_MODEL_CACHE` – Hugging Face snapshot cache path (default `data/model-cache`).

When the database is empty a hardened admin account is created (`admin` with a generated password logged once). Change it immediately via the console.

## SQLite Schema

The bundled database contains the following tables:

- `users` – administrator credentials hashed with Argon2id.
- `admin_sessions` – hashed session tokens with expirations.
- `api_keys` – hashed client API keys with aggregate token counters.
- `token_usage` – per-request prompt/completion accounting.
- `model_downloads` – Hugging Face snapshots with size metadata.
- `provider_credentials` – encrypted provider tokens (ChaCha20-Poly1305).

All write operations execute inside `tokio::task::spawn_blocking` to avoid blocking the async runtime.

## Hugging Face Downloads

The admin UI exposes a "Download Model" form. Tokens are stored using the credential vault and decrypted only during `HuggingFaceDownloader::download_repo`. Snapshots are cached to `LLMSERVER_MODEL_CACHE` and recorded in the database with size information derived from `walkdir`.

## Model Lifecycle

The `ModelManager` orchestrates actors for three model classes:

- **LLM** – launches `SimpleRkLLM` instances.
- **ASR** – launches `SimpleASR` speech-to-text processors.
- **TTS** – launches `SimpleToneTts`, a deterministic waveform synthesizer for audible confirmations.

Instances can be preloaded via CLI (`--model <repo> --instances <n>`) or created on demand from the admin console. Graceful shutdown broadcasts `ShutdownMessages` to every actor when the server terminates.

## Authentication Flow

1. Administrators authenticate at `/admin/login` (cookies scoped to `/admin`).
2. API consumers supply `X-API-Key` or `Authorization: Bearer <token>` headers.
3. Successful requests record token usage in `token_usage` for billing/tracking.

## REST Endpoints

- `POST /v1/chat/completions` – text completions with streaming SSE and usage tracking.
- `POST /v1/audio/transcriptions` – SenseVoice-powered transcription, returns JSON text.
- `POST /v1/audio/speech` – waveform TTS returning base64-encoded WAV data.
- `GET /admin` – dashboard with live state, download history, and management forms.

Every endpoint is documented in the generated OpenAPI spec served under `/swagger-ui/`.

## Port & Deployment

The HTTP server now binds to `0.0.0.0:8443`. Terminate TLS at a reverse proxy or enable TLS termination in the deployment environment for production. Use the admin console only over secure transport.

# Administration Console and Runtime Services

This release adds a secure administration surface to the LLM server. The highlights below explain how to operate the new components and how data is persisted.

## Storage and Persistence

* **Database** – The server stores configuration and auditing data in SQLite.  The location is configurable through `LLMSERVER_DATABASE_URL` (defaults to `sqlite://data/app.db`).
  * `users` – Administrative accounts authenticated in the web UI.
  * `api_keys` – API credentials used by clients.  Hashed secrets and usage counters are stored here.
  * `token_usage_events` – Per-request audit trail for token consumption.
  * `models` – Metadata for text, speech, and TTS models managed by the control plane.
  * `huggingface_tokens` – Encrypted Hugging Face access tokens (requires `LLMSERVER_ENCRYPTION_KEY`).
* **Migrations** – Schema migrations are bundled with the application (`migrations/20250101000000_init.sql`). They run automatically on startup.

## Required Environment

| Variable | Purpose |
| --- | --- |
| `LLMSERVER_DATABASE_URL` | Location of the SQLite database. |
| `LLMSERVER_ENCRYPTION_KEY` | 64-character hex string used to encrypt Hugging Face tokens. |
| `LLMSERVER_SESSION_KEY` | 128-character hex string (64 bytes) securing session cookies. |
| `LLMSERVER_HF_CACHE` | Optional custom Hugging Face cache directory. |
| `LLMSERVER_BIND_ADDRESS` | Optional bind address (defaults to `0.0.0.0:8443`). |

## Hugging Face Integration

* Use the admin UI to store a token in the encrypted vault (`/admin/api/hf-tokens`).
* Model downloads and inference requests rely on the stored token.  Downloads use the `hf-hub` client and land inside the configured cache directory.
* The text-to-speech endpoint calls the Hugging Face Inference API (`https://api-inference.huggingface.co/models/<repo>`).  Audio responses are streamed back to clients.

## API Keys and Usage Accounting

* Keys are generated in the admin UI or through `POST /admin/api/api-keys`.  Only the last four characters are persisted.  The raw key is shown once upon creation.
* Every request to `/v1/chat/completions`, `/v1/audio/transcriptions`, and `/v1/audio/speech` must supply a key via `Authorization: Bearer <key>` or `X-API-Key`.
* Token usage is measured with `tiktoken-rs` (`cl100k_base`).
  * Prompt tokens are counted for all requests.
  * Completion tokens are recorded for text responses; TTS calls track prompt tokens only.
  * Usage is persisted in `api_keys` and individual events are written to `token_usage_events`.
* Optional per-key quotas (`token_limit`) enforce cumulative token ceilings.

## Model Lifecycle

* Register models through `/admin/api/models` (or the admin UI).  The server writes JSON configs to `assets/config/<name>.json`.
* Downloads pull the latest snapshot from Hugging Face and update the model record.
* `LLM` models are launched as Actix actors (`SimpleRkLLM`) and tracked by the `ModelManager` for request routing.
* `TTS` models rely on remote inference; starting a model toggles status but does not spawn local actors.
* Existing running models are resumed automatically when the server boots.

## Web Administration UI

* Accessible at `/admin`.  Provides login, API key management, model orchestration, and Hugging Face token management.
* Relies on session cookies issued by the Actix session middleware.

## Security Notes

* All secrets (API keys, Hugging Face tokens) are stored using strong cryptography (Argon2 hashing and AES-GCM encryption).
* Administrative endpoints require authentication.  Client endpoints require API keys.
* Server binds to `0.0.0.0:8443` by default to avoid insecure development ports.

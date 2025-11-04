# PostgresML Administration

This document describes how `llmserver-rs` integrates with [PostgresML](https://postgresml.org/) (PGML) to manage Retrieval-Augmented Generation (RAG) pipelines directly from the server's Admin Dashboard.

## Overview

* A dedicated **PGML repository** stores pipeline metadata in Postgres, using the `admin_rag_models` table.
* The API automatically enables the Admin Dashboard when the `DATABASE_URL` environment variable is present.
* Operators can load HuggingFace or Supabase models as PGML pipelines without leaving the dashboard.
* Supabase is the preferred managed backend. `docker-compose.supabase.yml` provisions a Supabase Postgres service and installs PGML via the [`eyshoit-commits/pgml`](https://github.com/eyshoit-commits/pgml) image.
* The implementation prefers a lightweight PostgresML instance derived from the [`docker-rust-postgres`](https://github.com/docker/docker-rust-postgres/) template. If a self-managed deployment is not possible, Supabase can be used as an alternative backend.

## Database Schema

The PGML repository automatically creates and maintains the following objects:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgml;

CREATE TABLE IF NOT EXISTS admin_rag_models (
    pipeline_name TEXT PRIMARY KEY,
    model_uri TEXT NOT NULL,
    task TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_status TEXT NOT NULL DEFAULT 'registered',
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

`pgml.load_model` is executed each time a pipeline is created or updated. When removal is requested the system attempts to call `pgml.drop_model` and then removes the metadata from `admin_rag_models`.

## Running a local PostgresML instance

A minimal PGML server can be started via Docker Compose. Copy `.env.example` to `.env` or export the environment variable before launching:

```bash
export PGML_POSTGRES_PASSWORD='pGml-Admin#2025!Secure'
docker compose -f docker-compose.pgml.yml up -d
```

* Postgres listens on `localhost:6543` (avoiding insecure defaults).
* Credentials: `pgml_admin` / `${PGML_POSTGRES_PASSWORD}`.
* Data is persisted under the `pgml-data` Docker volume.

Once the container is healthy set the application environment variable:

```bash
export DATABASE_URL="postgresql://pgml_admin:${PGML_POSTGRES_PASSWORD}@localhost:6543/pgml"
```

Start `llmserver-rs` afterwards; the Admin Dashboard will become available at `http://localhost:8443/admin`.

## Supabase compose stack

Use the bundled Supabase compose file to run a TLS-enabled Postgres service together with the PGML migrator:

```bash
export SUPABASE_DB_PASSWORD='Supabase#RotateMe2025!'
docker compose -f docker-compose.supabase.yml up -d

export DATABASE_URL="postgresql://postgres:${SUPABASE_DB_PASSWORD}@localhost:7654/postgres"
export PGML_TLS_MODE=require
```

The `pgml-extension` service depends on the Supabase database. It issues the PGML migrations through the containerised CLI from `eyshoit-commits/pgml` and then idles, ensuring the extension remains installed while the stack is running.

## Supabase alternative

If you prefer Supabase's hosted control plane instead of the bundled compose stack:
## Supabase alternative

If Docker-based PostgresML is not an option:

1. Create a Supabase project and enable the `vector` extension (`Database → Extensions`).
2. Within the Supabase SQL editor run:

   ```sql
   CREATE EXTENSION IF NOT EXISTS pgml;
   ```

3. Retrieve the connection string from the project settings and map it to `DATABASE_URL`.
4. Set `PGML_TLS_MODE=require` because Supabase only exposes TLS-enabled Postgres endpoints.
5. Ensure the service role key is stored securely (for example via the hosting platform's secret manager).

The Admin Dashboard honours the TLS mode automatically. When `PGML_TLS_MODE=require` is present all queries, migrations, and PGML calls are executed over a `rustls`-backed encrypted channel so Supabase compatibility is ensured without reverse proxies or stunnel sidecars.

Supabase exposes a managed Postgres instance that is compatible with the dashboard workflows. You can still run `docker-compose.pgml.yml` locally for development and switch to Supabase in production.

## Admin Dashboard & Authentication
4. Ensure the service role key is stored securely (for example via the hosting platform's secret manager).

Supabase exposes a managed Postgres instance that is compatible with the dashboard workflows. You can still run `docker-compose.pgml.yml` locally for development and switch to Supabase in production.

## Admin Dashboard

Navigate to `/admin` to access the SPA-like dashboard:

* **Register / Update Pipeline** – supplies `pipeline_name`, `model_uri`, `task`, `collection_name`, and optional JSON metadata.
* **Delete Pipeline** – removes the metadata and calls `pgml.drop_model` when available.
* **Live Status** – the most recent `pgml.load_model` result is displayed, including error messages.
* **Token-based protection** – when `ADMIN_API_TOKEN` is configured, the dashboard prompts for the bearer token and stores it in `localStorage`. All API calls automatically include the `Authorization: Bearer` header.

The UI communicates over the `/admin/api` REST endpoints, which are also described in Swagger (`/swagger-ui`). Leave `ADMIN_API_TOKEN` unset for trusted air-gapped environments; production deployments must set a strong token.

## TinyLlama startup model

The default CLI argument loads `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` from `assets/config/tinyllama-1.1b-chat.json`. The Q8_0 variant keeps disk usage low while allowing you to validate PGML ingestion quickly. Pass `--model-name <repo>` to override the startup checkpoint.

## Security Recommendations

* Store `PGML_POSTGRES_PASSWORD`, `SUPABASE_DB_PASSWORD`, `ADMIN_API_TOKEN`, and `DATABASE_URL` in a vault or the orchestrator's secret management facility.

The UI communicates over the `/admin/api` REST endpoints, which are also described in Swagger (`/swagger-ui`).

## Security Recommendations

* Store `PGML_POSTGRES_PASSWORD` and `DATABASE_URL` in a vault or the orchestrator's secret management facility.
* Restrict network access to port `6543` (or the custom port chosen for Postgres) to trusted hosts.
* If Supabase is used, rotate service role keys frequently and enable row-level security policies where appropriate.
* Apply TLS termination in front of `llmserver-rs` when exposing the Admin Dashboard to the public internet.

With these steps in place you can load and maintain PGML-powered RAG models directly from the Admin Dashboard.

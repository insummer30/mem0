# Mem0 REST API Server

Mem0 provides a REST API server (written using FastAPI). Users can perform all operations through REST endpoints. The API also includes OpenAPI documentation, accessible at `/docs` when the server is running.

## Features

- **Create memories:** Create memories based on messages for a user, agent, or run.
- **Retrieve memories:** Get all memories for a given user, agent, or run.
- **Search memories:** Search stored memories based on a query.
- **Update memories:** Update an existing memory.
- **Delete memories:** Delete a specific memory or all memories for a user, agent, or run.
- **Reset memories:** Reset all memories for a user, agent, or run.
- **OpenAPI Documentation:** Accessible via `/docs` endpoint.

## Running the server

Follow the instructions in the [docs](https://docs.mem0.ai/open-source/features/rest-api) to run the server.

## Startup config files

By default, the server starts with [`configs/config.json`](./configs/config.json). To boot with a different Mem0 config, set `MEM0_CONFIG_PATH` to another JSON file. The file is loaded during startup and passed directly to `Memory.from_config(...)`.

If `MEM0_CONFIG_PATH` is unset, the server still uses `server/configs/config.json`.

## Environment files

When the server starts, it loads environment variables in this order:

1. existing exported environment variables
2. `server/.env`
3. repo-root `.env` only if `server/.env` does not exist

That means `server/.env` is the canonical place for local REST server credentials such as `GOOGLE_API_KEY`, while exported variables still take precedence.

## Local Gemini Flash + Qdrant example

A sample startup config is included at [`configs/config.json`](./configs/config.json).

1. Start Qdrant locally.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

2. Create `server/.env` and set the Gemini API key. `MEM0_CONFIG_PATH` is optional because it already defaults to `server/configs/config.json`.

```bash
cp server/.env.example server/.env
```

```env
GOOGLE_API_KEY=your-google-api-key
```

3. Start the server from the repo root.

```bash
uv run --with fastapi --with python-dotenv --with uvicorn --with google-genai uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Open the API docs.

```text
http://localhost:8000/docs
```

Notes:

- The sample config relies on `GOOGLE_API_KEY` from the environment, so the JSON file does not need to contain credentials.
- If `GOOGLE_API_KEY` is already exported in your shell, it overrides the value in `server/.env`.
- If you want to use a different startup config file, set `MEM0_CONFIG_PATH` explicitly.
- `embedding_model_dims` is set to `768` to match `models/gemini-embedding-001`.
- If you run the server itself inside Docker, change the Qdrant host in the JSON file from `localhost` to a host reachable from the container, such as `host.docker.internal`.

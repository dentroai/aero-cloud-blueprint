# Aero Cloud Blueprint

A ready-to-use Docker setup for Retrieval Augmented Generation (RAG) projects that includes all necessary components running in containers.

## What's Included

- **vLLM**: Large language model server
- **Ollama**: Embedding model server
- **Qdrant**: Vector database
- **Flowise**: No-code chat flow builder (custom fork)
- **Langfuse**: Tracing for chatbot interactions

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Sufficient RAM and GPU VRAM for the selected models (varies greatly, see model recommendations below).
- Sufficient free disk space for Docker images and model downloads (can range from ~20GB to hundreds of GB).

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/aero-cloud-blueprint.git # Replace with the actual repo URL if known
   cd aero-cloud-blueprint
   ```

2. **Configure environment variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Now, edit the `.env` file with your specific settings:
   - **Required:** Set `HUGGING_FACE_HUB_TOKEN` if you plan to use gated models like Llama or Gemma V2 from Hugging Face. You can get a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - **Required:** Choose your `VLLM_MODEL_NAME`. Most models from Hugging Face Hub should work. 
     - For **testing** on systems with no or lower VRAM (e.g., 2GB), try a smaller model like `google/gemma-3-1b-it`.
     - For **powerful GPUs** (e.g., 80GB+ VRAM), you could use large models like `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`.
     - **Recommendation:** Consider using models quantized with the **AWQ** (Activation-aware Weight Quantization) method (like the Qwen example above) as they offer a good balance between performance and reduced VRAM/disk space requirements compared to their unquantized counterparts. Search for `[model-name]-awq` on Hugging Face.
     - **Note:** The required VRAM also depends significantly on the `VLLM_MAX_MODEL_LEN` (context window size) set in your `.env` file. Larger context windows require more VRAM.
   - **Required:** Set a secure `FLOWISE_USERNAME` and `FLOWISE_PASSWORD` for accessing the Flowise UI.
   - **Recommended:** Review and change the default `LANGFUSE_*` passwords and secrets in the `.env` file, especially `LANGFUSE_DB_PASSWORD`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_CLICKHOUSE_PASSWORD`, `LANGFUSE_MINIO_PASSWORD`, `LANGFUSE_REDIS_PASSWORD`, and `LANGFUSE_NEXTAUTH_SECRET`. Generate a secure `LANGFUSE_ENCRYPTION_KEY` using `openssl rand -hex 32`.

3. **Start the services**:
   Run the following command in your terminal from the project's root directory:
   ```bash
   docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml up -d
   ```
   This command downloads the necessary images (if not already present), builds the custom ones (Ollama, Flowise), and starts all the containers in the background (`-d`). The first startup might take some time, especially for downloading large models.

4. **Access the services**:
   Once the containers are running, you can access the web interfaces:
   - **Flowise**: [http://localhost:3000](http://localhost:3000) (Use the username and password you set in `.env`)
   - **Langfuse**: [http://localhost:3001](http://localhost:3001)
   - **Qdrant UI (Optional)**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
   - **Minio UI (Optional)**: [http://localhost:9090](http://localhost:9090) (Login with user `minio` and the `LANGFUSE_MINIO_PASSWORD` from `.env`)

## Connecting Services in Flowise

When building chatflows in the Flowise UI ([http://localhost:3000](http://localhost:3000)), you'll need to connect to the other running services. Because all containers are on the same Docker network (`rag_network`), they can communicate using their service names as hostnames.

Here are the typical settings:

| Service to Connect | Flowise Node          | Key Parameter(s)         | Value(s)                                      | Notes                                                                 |
|--------------------|-----------------------|--------------------------|-----------------------------------------------|-----------------------------------------------------------------------|
| **vLLM (LLM)**     | `ChatOpenAI Custom`   | `Base Path`              | `http://vllm:8000/v1`                         | Use the container name `vllm` and port `8000`.                        |
|                    |                       | `Model Name`             | `vllm-model`                                  | This matches `VLLM_SERVED_MODEL_NAME` in your `.env` file.            |
|                    |                       | `OpenAI API Key`         | `1234XYZ` (or any other dummy value)                   | The vLLM OpenAI endpoint doesn't require a key by default, but Flowise expects one.          |
| **Ollama (Embeddings)**| `Ollama Embeddings` | `Base URL`               | `http://ollama:11434`                         | Use the container name `ollama` and port `11434`.                     |
|                    |                       | `Model Name`             | `${OLLAMA_EMBEDDING_MODEL}`                   | Use the embedding model name specified in your `.env` file.         |
| **Qdrant (Vector DB)**| `Qdrant`            | `Qdrant Server URL`      | `http://qdrant:6333`                          | Use the container name `qdrant` and port `6333`.                      |
|                    |                       | `Collection Name`        | *Your desired collection name*                |                                                                       |
|                    |                       | `Qdrant API Key`              | `1234XYZ` (or any other dummy value)                   | The qdrant endpoint doesn't require a key by default, but Flowise expects one.
| **Langfuse (Tracing)**| `Langfuse`          | `Endpoint`               | `http://langfuse-web:3000`                    | Use the container name `langfuse-web` and port `3000`.                |
|                    |                       | `Secret Key` / `Public Key`| *Get these from the Langfuse UI* ([http://localhost:3001](http://localhost:3001)) | Go to Project Settings -> API Keys in Langfuse to generate keys. |

**Important:** When using these URLs within Flowise, Docker's internal networking resolves the service names (like `vllm`, `ollama`, `qdrant`, `langfuse-web`) to the correct container IP addresses. Do not use `localhost` for these internal connections.

## Component Details

*   **vLLM**: High-throughput LLM serving engine using the OpenAI API format. Configuration is managed via the `.env` file (`VLLM_*` variables). Runs on GPU. The VRAM needed depends heavily on the chosen model size and the configured maximum context length (`VLLM_MAX_MODEL_LEN`).
*   **Ollama**: Serves embedding models (and can also serve LLMs, though vLLM is used here for the main LLM). The Dockerfile pre-pulls the embedding model specified in `.env` (`OLLAMA_EMBEDDING_MODEL`). You can change this to any other model available in the [Ollama Library](https://ollama.com/library). **Crucially**, ensure you use the *same* embedding model when adding data to Qdrant (indexing/upserting) and when querying it later in your Flowise flow.
*   **Qdrant**: Fast vector database for storing and searching embeddings generated by Ollama. Data is persisted in the `qdrant_data` volume.
*   **Flowise**: A visual tool for building RAG applications. This setup uses a specific fork (`dentroai/Flowise`). Configuration and data are stored in `flowise_data*` volumes.
    *   **Custom Feature (Audit Trail)**: This fork has a modification compared to the standard Flowise. Each time you save a chatflow in the UI, besides saving to the main database, it also saves a JSON representation of the flow to a dedicated volume. These JSON files are stored in a folder named after the chatflow's ID, and the filename is the UTC timestamp of when the save occurred (e.g., `/data/<flow-id>/<timestamp>.json`). This serves as an audit trail or version history of your flows. The base path for this storage inside the container is defined by the `FLOWISE_AUDIT_TRAIL_STORAGE_PATH` environment variable (defaulting to `/data/`) and is mapped to the `flowise_data_audit` Docker volume.
*   **Langfuse**: An open-source observability and analytics platform for LLM applications. It provides detailed tracing of your Flowise interactions. It consists of several containers (`langfuse-web`, `langfuse-worker`, `postgres`, `redis`, `clickhouse`, `minio`) for its backend, database, cache, analytics DB, and object storage. Data is persisted in `langfuse_*` volumes.

## Document Processing Pipeline

The repository includes a document processing pipeline script to help you ingest documents into your RAG system:

```bash
cd python-scripts
python3 etl.py \
    --docs-folder "../documents/" \
    --images-output "../documents/tmp/" \
    --vllm-model "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    --ollama-model "snowflake-arctic-embed2" \
    --qdrant-collection "document_embeddings"
```

This script processes documents from the specified folder, generates embeddings using the configured models, and stores them in the Qdrant vector database for later retrieval.

## Managing the Services

*   **View Logs**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs -f [service_name]` (e.g., `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs -f vllm`). Omit `[service_name]` to see logs for all services.
*   **Stop Services**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml down`. This stops and removes the containers but keeps the volumes (data).
*   **Stop and Remove Data**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml down -v`. **Warning**: This deletes all data stored in the volumes (vector embeddings, Flowise chats, Langfuse traces, etc.).
*   **Rebuild a specific service**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build [service_name]` (e.g., `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build flowise` if you modify its source).
*   **Restart services**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml restart`.

## Troubleshooting

*   **GPU Not Detected / CUDA Errors**:
    *   Verify NVIDIA drivers are installed on the host machine.
    *   Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed and Docker is configured to use it.
    *   Check `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs vllm` and `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs ollama` for specific errors.
*   **Container Exits Immediately**: Check logs using `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs [service_name]`. Common issues include incorrect `.env` variables, insufficient resources (RAM/VRAM), or port conflicts.
*   **Model Download Issues (vLLM/Ollama)**:
    *   Ensure your `HUGGING_FACE_HUB_TOKEN` in `.env` is correct and has access to the model if it's gated.
    *   Check internet connectivity from within the container if possible (`docker exec -it [container_name] ping google.com`).
    *   Ensure sufficient disk space in `hf_cache` and `ollama_data` volumes.
*   **Flowise Connection Errors**: Double-check the URLs used in Flowise nodes – ensure they use the service names (e.g., `http://vllm:8000/v1`) and not `localhost`. Verify the target service is running (`docker ps`).
*   **Langfuse Errors**: Check logs for all `langfuse-*` services and their dependencies (`postgres`, `redis`, `clickhouse`, `minio`). Ensure passwords in `.env` match those used by the services.

## Customization

*   **Change Models**: Modify `VLLM_MODEL_NAME` and `OLLAMA_EMBEDDING_MODEL` in the `.env` file. Remember to restart the respective services (`docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml restart vllm ollama`). Ensure consistency if changing the embedding model (see Ollama details above).
*   **Adjust Resources**: Modify GPU memory utilization (`VLLM_GPU_MEM_UTIL`) or max model length (`VLLM_MAX_MODEL_LEN`) in `.env`. For CPU/RAM limits (less common for these GPU-focused services), you would adjust the `deploy.resources` section in `docker-compose.yaml`.
*   **Update Flowise/Langfuse**: Change the image tags or build contexts in `docker-compose.yaml` or `langfuse/docker-compose.langfuse.yaml`. Remember to run `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build [service_name]` and `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml up -d`
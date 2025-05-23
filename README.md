# Aero Cloud Blueprint

A ready-to-use Docker setup for Retrieval Augmented Generation (RAG) projects that includes all necessary components running in containers.

## What's Included

- **vLLM**: Large language model server
- **Ollama**: Embedding model server
- **PostgreSQL with pgvector**: Vector database
- **Flowise**: No-code chat flow builder (custom fork)
- **Langfuse**: Tracing for chatbot interactions
- **Aero Chat**: Next.js application with MSAL authentication

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
   Create a `.env` file with your specific settings:
   ```bash
   # PostgreSQL Configuration
   POSTGRES_PORT=5432
   POSTGRES_DB=rag_db
   POSTGRES_USER=rag_user
   POSTGRES_PASSWORD=your_secure_postgres_password

   # Hugging Face Configuration
   HUGGING_FACE_HUB_TOKEN=your_hugging_face_token
   HF_HUB_ENABLE_HF_TRANSFER=false

   # vLLM LLM Configuration
   VLLM_LLM_PORT=8000
   VLLM_LLM_MODEL_NAME=microsoft/DialoGPT-medium
   VLLM_LLM_MAX_MODEL_LEN=8192
   VLLM_LLM_GPU_MEM_UTIL=0.9
   VLLM_LLM_SERVED_MODEL_NAME=vllm-llm-model
   VLLM_TENSOR_PARALLEL_SIZE=1

   # vLLM Text Embedding Configuration
   VLLM_TEXT_EMBEDDING_PORT=8001
   VLLM_TEXT_EMBEDDING_MODEL_NAME=Snowflake/snowflake-arctic-embed-l-v2.0
   VLLM_TEXT_EMBEDDING_SERVED_NAME=text-embedding-model
   VLLM_TEXT_EMBEDDING_GPU_MEM_UTIL=0.9
   VLLM_TEXT_EMBEDDING_MAX_LEN=4096
   VLLM_TEXT_EMBEDDING_TRUST_REMOTE_CODE=true

   # vLLM Image Embedding Configuration
   VLLM_IMAGE_EMBEDDING_PORT=8002
   VLLM_IMAGE_EMBEDDING_MODEL_NAME=your_image_embedding_model
   VLLM_IMAGE_EMBEDDING_SERVED_NAME=image-embedding-model
   VLLM_IMAGE_EMBEDDING_GPU_MEM_UTIL=0.9
   VLLM_IMAGE_EMBEDDING_TRUST_REMOTE_CODE=false

   # Flowise Configuration
   FLOWISE_PORT=3002
   FLOWISE_DATABASE_PATH=/root/.flowise
   FLOWISE_APIKEY_PATH=/root/.flowise
   FLOWISE_SECRETKEY_PATH=/root/.flowise
   FLOWISE_LOG_PATH=/root/.flowise/logs
   FLOWISE_BLOB_STORAGE_PATH=/root/.flowise/storage
   FLOWISE_AUDIT_TRAIL_STORAGE_PATH=/data/
   FLOWISE_DISABLE_TELEMETRY=true
   FLOWISE_USERNAME=your_flowise_username
   FLOWISE_PASSWORD=your_flowise_password
   FLOWISE_CORS_ORIGINS=*
   FLOWISE_IFRAME_ORIGINS=*

   # Aero Chat Configuration
   AERO_CHAT_PORT=3000
   GITHUB_TOKEN=your_github_personal_access_token
   AERO_CHAT_REPO_OWNER=your-github-username
   AERO_CHAT_REPO_NAME=your-private-repo-name
   AERO_CHAT_REPO_BRANCH=main
   NEXT_PUBLIC_MSAL_CLIENT_ID=your_msal_client_id
   NEXT_PUBLIC_MSAL_TENANT_ID=your_msal_tenant_id
   NEXT_PUBLIC_BASE_URL=http://localhost:${AERO_CHAT_PORT:-3000}

   # Langfuse Configuration (if using)
   LANGFUSE_PORT=3001
   LANGFUSE_DB_PASSWORD=your_langfuse_db_password
   LANGFUSE_SALT=your_salt
   LANGFUSE_ENCRYPTION_KEY=your_encryption_key
   LANGFUSE_CLICKHOUSE_PASSWORD=your_clickhouse_password
   LANGFUSE_MINIO_PASSWORD=your_minio_password
   LANGFUSE_REDIS_PASSWORD=your_redis_password
   LANGFUSE_NEXTAUTH_SECRET=your_nextauth_secret
   ```

   **Key Environment Variables:**
   - **Required:** Set `HUGGING_FACE_HUB_TOKEN` if you plan to use gated models like Llama or Gemma V2 from Hugging Face. You can get a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - **Required:** Choose your `VLLM_MODEL_NAME`. Most models from Hugging Face Hub should work. 
     - For **testing** on systems with no or lower VRAM (e.g., 2GB), try a smaller model like `google/gemma-3-1b-it`.
     - For **powerful GPUs** (e.g., 80GB+ VRAM), you could use large models like `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`.
     - **Recommendation:** Consider using models quantized with the **AWQ** (Activation-aware Weight Quantization) method (like the Qwen example above) as they offer a good balance between performance and reduced VRAM/disk space requirements compared to their unquantized counterparts. Search for `[model-name]-awq` on Hugging Face.
     - **Note:** The required VRAM also depends significantly on the `VLLM_MAX_MODEL_LEN` (context window size) set in your `.env` file. Larger context windows require more VRAM.
   - **Required:** Set a secure `FLOWISE_USERNAME` and `FLOWISE_PASSWORD` for accessing the Flowise UI.
   - **Required:** Set `POSTGRES_PASSWORD` for the main RAG PostgreSQL database.
   - **Required:** Set `GITHUB_TOKEN` to your GitHub Personal Access Token with `repo` scope.
   - **Required:** Configure `AERO_CHAT_REPO_OWNER`, `AERO_CHAT_REPO_NAME`, and `AERO_CHAT_REPO_BRANCH` for your private repository.
   - **Required:** Configure MSAL authentication variables for Aero Chat.
   - **Recommended:** Review and change the default `LANGFUSE_*` passwords and secrets in the `.env` file, especially `LANGFUSE_DB_PASSWORD`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_CLICKHOUSE_PASSWORD`, `LANGFUSE_MINIO_PASSWORD`, `LANGFUSE_REDIS_PASSWORD`, and `LANGFUSE_NEXTAUTH_SECRET`. Generate a secure `LANGFUSE_ENCRYPTION_KEY` using `openssl rand -hex 32`.

3. **Start the services**:
   Run the following command in your terminal from the project's root directory:
   ```bash
   docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml up -d
   ```
   This command downloads the necessary images (if not already present), builds the custom ones (Ollama, Flowise), and starts all the containers in the background (`-d`). The first startup might take some time, especially for downloading large models.

4. **Access the services**:
   Once the containers are running, you can access the web interfaces:
   - **Aero Chat**: [http://localhost:${AERO_CHAT_PORT:-3000}](http://localhost:${AERO_CHAT_PORT:-3000}) (Next.js application with MSAL authentication)
   - **Langfuse**: [http://localhost:${LANGFUSE_PORT:-3001}](http://localhost:${LANGFUSE_PORT:-3001})
   - **Flowise**: [http://localhost:${FLOWISE_PORT:-3002}](http://localhost:${FLOWISE_PORT:-3002}) (Use the username and password you set in `.env`)
   - **Minio UI (Optional, used by Langfuse)**: [http://localhost:9090](http://localhost:9090) (Login with user `minio` and the `LANGFUSE_MINIO_PASSWORD` from `.env`)
   - You can connect to the main RAG PostgreSQL database (service name `postgres`) using a PostgreSQL client on `localhost:5432` (or the `POSTGRES_PORT` you set) with the credentials from your `.env` file.

## Connecting Services in Flowise

When building chatflows in the Flowise UI ([http://localhost:${FLOWISE_PORT:-3002}](http://localhost:${FLOWISE_PORT:-3002})), you'll need to connect to the other running services. Because all containers are on the same Docker network (`rag_network`), they can communicate using their service names as hostnames.

Here are the typical settings:

| Service to Connect | Flowise Node          | Key Parameter(s)         | Value(s)                                      | Notes                                                                 |
|--------------------|-----------------------|--------------------------|-----------------------------------------------|-----------------------------------------------------------------------|
| **vLLM (LLM)**     | `ChatOpenAI Custom`   | `Base Path`              | `http://vllm:8000/v1`                         | Use the container name `vllm` and port `8000`.                        |
|                    |                       | `Model Name`             | `vllm-model`                                  | This matches `VLLM_SERVED_MODEL_NAME` in your `.env` file.            |
|                    |                       | `OpenAI API Key`         | `1234XYZ` (or any other dummy value)                   | The vLLM OpenAI endpoint doesn't require a key by default, but Flowise expects one.          |
| **Ollama (Embeddings)**| `Ollama Embeddings` | `Base URL`               | `http://ollama:11434`                         | Use the container name `ollama` and port `11434`.                     |
|                    |                       | `Model Name`             | `${OLLAMA_EMBEDDING_MODEL}`                   | Use the embedding model name specified in your `.env` file.         |
| **PostgreSQL (Vector DB)**| `PostgreSQL (Vector Store)` | `Postgres Connection String` | `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}` | Use service name `postgres`. Ensure credentials match your `.env`. |
|                    |                       | `Table Name`             | `document_chunks`                             | This is the default table name used by the ETL script.              |
|                    |                       | `Content Column Name`    | `text`                                        |                                                                       |
|                    |                       | `Embedding Column Name`  | `embedding`                                   |                                                                       |
|                    |                       | `Metadata Column Name(s)`| `file_name, page_number, file_path` (comma-separated) | Add other metadata columns as needed.                               |
| **Langfuse (Tracing)**| `Langfuse`          | `Endpoint`               | `http://langfuse-web:3000`                    | Use the container name `langfuse-web` and port `3000`.                |
|                    |                       | `Secret Key` / `Public Key`| *Get these from the Langfuse UI* ([http://localhost:${LANGFUSE_PORT:-3001}](http://localhost:${LANGFUSE_PORT:-3001})) | Go to Project Settings -> API Keys in Langfuse to generate keys. |

**Important:** When using these URLs within Flowise, Docker's internal networking resolves the service names (like `vllm`, `ollama`, `postgres`, `langfuse-web`) to the correct container IP addresses. Do not use `localhost` for these internal connections.

## Component Details

*   **Aero Chat**: A Next.js application with Microsoft Authentication Library (MSAL) integration for secure user authentication. Built using Bun runtime for optimal performance. Features include:
    *   **Database**: Uses SQLite with Prisma ORM for data persistence
    *   **Authentication**: MSAL (Microsoft Authentication Library) for Azure AD integration
    *   **Build Process**: Multi-stage Docker build with standalone output for production efficiency
    *   **Data Persistence**: SQLite database stored in `aero_chat_data` volume
*   **vLLM**: High-throughput LLM serving engine using the OpenAI API format. Configuration is managed via the `.env` file (`VLLM_*` variables). Runs on GPU. The VRAM needed depends heavily on the chosen model size and the configured maximum context length (`VLLM_MAX_MODEL_LEN`).
*   **Ollama**: Serves embedding models (and can also serve LLMs, though vLLM is used here for the main LLM). The Dockerfile pre-pulls the embedding model specified in `.env` (`OLLAMA_EMBEDDING_MODEL`). You can change this to any other model available in the [Ollama Library](https://ollama.com/library). **Crucially**, ensure you use the *same* embedding model when adding data to PostgreSQL (indexing/upserting) and when querying it later in your Flowise flow.
*   **PostgreSQL with pgvector**: A powerful relational database extended with `pgvector` for storing and searching embeddings generated by Ollama. Data is persisted in the `postgres_data` volume. The ETL script creates a table named `document_chunks`.
*   **Flowise**: A visual tool for building RAG applications. This setup uses a specific fork (`dentroai/Flowise`). Configuration and data are stored in `flowise_data*` volumes.
    *   **Custom Feature (Audit Trail)**: This fork has a modification compared to the standard Flowise. Each time you save a chatflow in the UI, besides saving to the main database, it also saves a JSON representation of the flow to a dedicated volume. These JSON files are stored in a folder named after the chatflow's ID, and the filename is the UTC timestamp of when the save occurred (e.g., `/data/<flow-id>/<timestamp>.json`). This serves as an audit trail or version history of your flows. The base path for this storage inside the container is defined by the `FLOWISE_AUDIT_TRAIL_STORAGE_PATH` environment variable (defaulting to `/data/`) and is mapped to the `flowise_data_audit` Docker volume.
*   **Langfuse**: An open-source observability and analytics platform for LLM applications. It provides detailed tracing of your Flowise interactions. It consists of several containers (`langfuse-web`, `langfuse-worker`, `langfuse_postgres`, `redis`, `clickhouse`, `minio`) for its backend, database, cache, analytics DB, and object storage. Data is persisted in `langfuse_*` volumes.

## Document Processing Pipeline

The repository includes a document processing pipeline script to help you ingest documents into your RAG system:

```bash
# Ensure your .env file has POSTGRES_PASSWORD set, or set it in your environment:
# export POSTGRES_PASSWORD="your_postgres_password_here"

cd python-scripts
python3 etl.py \
    --docs-folder "../documents/" \
    --images-output "../documents/tmp/" \
    --ollama-model "snowflake-arctic-embed2" \
    --postgres-host "localhost" \
    --postgres-port 5432 \
    --postgres-db "rag_db" \
    --postgres-user "rag_user" \
    --postgres-password "your_strong_postgres_password" \
    --ocr-language "de" \
    --max-workers 1
```

This script processes documents from the specified folder, generates embeddings using the configured models, and stores them (along with the page text and image) in the PostgreSQL database for later retrieval. 
The script will use the `POSTGRES_PASSWORD` from your environment or the `.env` file. Other PostgreSQL connection parameters default to the values expected when running alongside the Docker setup (e.g., host `postgres`). You can override them with CLI arguments if needed.

## Managing the Services

*   **View Logs**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs -f [service_name]` (e.g., `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml logs -f vllm`). Omit `[service_name]` to see logs for all services.
*   **Stop Services**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml down`. This stops and removes the containers but keeps the volumes (data).
*   **Stop and Remove Data**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml down -v`. **Warning**: This deletes all data stored in the volumes (vector embeddings, Flowise chats, Langfuse traces, etc.).
*   **Rebuild a specific service**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build [service_name]` (e.g., `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build flowise` if you modify its source).
*   **Restart services**: `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml restart`.

## Building from Private GitHub Repository

The Aero Chat service is configured to build directly from a private GitHub repository using GitHub Personal Access Token authentication.

### Setup Instructions

1. **Generate a GitHub Personal Access Token**:
   - Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
   - Click "Generate new token (classic)"
   - Select the `repo` scope (full control of private repositories)
   - Copy the generated token

2. **Configure Environment Variables**:
   Set these variables in your `.env` file:
   ```bash
   GITHUB_TOKEN=ghp_your_personal_access_token_here
   AERO_CHAT_REPO_OWNER=your-github-username
   AERO_CHAT_REPO_NAME=your-private-repo-name
   AERO_CHAT_REPO_BRANCH=main  # or your target branch
   ```

3. **Repository URL Format**:
   The docker-compose configuration automatically constructs the repository URL as:
   ```
   https://x-access-token:${GITHUB_TOKEN}@github.com/${AERO_CHAT_REPO_OWNER}/${AERO_CHAT_REPO_NAME}.git#${AERO_CHAT_REPO_BRANCH}
   ```
   Using `x-access-token` as the username explicitly tells Git that the provided `GITHUB_TOKEN` is an access token, preventing password prompts.

### Branch Selection
You can specify any branch by changing the `AERO_CHAT_REPO_BRANCH` environment variable. This is useful for:
- **Development**: `AERO_CHAT_REPO_BRANCH=develop`
- **Features**: `AERO_CHAT_REPO_BRANCH=feature/new-functionality`
- **Production**: `AERO_CHAT_REPO_BRANCH=main`

### Alternative Approaches

**Option 1: Local Clone** (if you prefer not to embed credentials):
1. Clone the repository locally: `git clone git@github.com:your-username/your-private-repo.git ../aero-chat`
2. Update the docker-compose.yaml context to: `context: ../aero-chat`

**Option 2: SSH with Docker Buildkit**:
```bash
# Build with SSH agent forwarding
DOCKER_BUILDKIT=1 docker compose build aero-chat --ssh default
```

**Security Note**: Be careful with tokens in environment files. Consider using Docker secrets or build-time secrets for production deployments. Never commit your `.env` file to version control.

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
*   **Aero Chat Build/Authentication Issues**:
    *   **Private repo access**: Ensure your GitHub token has `repo` scope and the repository URL includes authentication
    *   **MSAL configuration**: Verify `NEXT_PUBLIC_MSAL_CLIENT_ID` and `NEXT_PUBLIC_MSAL_TENANT_ID` are correctly set
    *   **Database errors**: Check that the `/app/data` volume has proper write permissions
    *   **Port conflicts**: If port 3000 is already in use, change `AERO_CHAT_PORT` in your `.env` file

## Customization

*   **Change Models**: Modify `VLLM_MODEL_NAME` and `OLLAMA_EMBEDDING_MODEL` in the `.env` file. Remember to restart the respective services (`docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml restart vllm ollama`). Ensure consistency if changing the embedding model (see Ollama details above).
*   **Adjust Resources**: Modify GPU memory utilization (`VLLM_GPU_MEM_UTIL`) or max model length (`VLLM_MAX_MODEL_LEN`) in `.env`. For CPU/RAM limits (less common for these GPU-focused services), you would adjust the `deploy.resources` section in `docker-compose.yaml`.
*   **Update Flowise/Langfuse**: Change the image tags or build contexts in `docker-compose.yaml` or `langfuse/docker-compose.langfuse.yaml`. Remember to run `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml build [service_name]` and `docker compose -f docker-compose.yaml -f langfuse/docker-compose.langfuse.yaml up -d`
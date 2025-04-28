# Use the official Ollama image as the base
FROM ollama/ollama:latest

# Define an argument for the embedding model name (mandatory)
ARG OLLAMA_EMBEDDING_MODEL

# Start ollama serve in background, wait, then pull the model.
# The serve process will stop once the RUN command finishes.
RUN ollama serve & \
    sleep 10 && \
    ollama pull ${OLLAMA_EMBEDDING_MODEL}

# Expose the default Ollama port
EXPOSE 11434 
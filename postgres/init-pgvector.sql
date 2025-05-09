-- postgres/init-flowise-schema.sql

-- Ensure the pgvector extension is enabled (idempotent)
-- This might be redundant if your init-pgvector.sql already does it,
-- but it's safe to include. Also ensure uuid-ossp is available if using UUID defaults.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- For uuid_generate_v4() if we want DB default

-- Create the 'documents' table that Flowise expects
-- Aligning with Flowise's auto-created 'documents_docstore'
-- YOUR_VECTOR_SIZE should be replaced with the actual dimension, e.g., 1024
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), -- Changed to UUID, added default
  "pageContent" TEXT, -- Changed column name to pageContent (quoted for case sensitivity if needed, though likely not here)
  metadata JSONB, -- Kept as metadata, Flowise uses this
  embedding VECTOR(1024) -- Replace 1024 with your actual embedding dimension. Kept explicit dimension.
  -- Removed image_data and timestamp to simplify and match Flowise's table
);

-- Create the 'match_documents' function that Flowise uses for searching
-- YOUR_VECTOR_SIZE should be replaced with the actual dimension, e.g., 1024
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1024), -- Replace 1024 with your actual embedding dimension
  match_count INT DEFAULT 5,
  filter JSONB DEFAULT '{}'
) RETURNS TABLE (
  id UUID, -- Changed to UUID
  "pageContent" TEXT, -- Changed column name
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d."pageContent", -- Changed column name
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents AS d
  WHERE (filter IS NULL OR d.metadata @> filter)
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- You might want to create an index on the embedding column for performance
-- The exact type of index (e.g., HNSW, IVFFlat) depends on your pgvector version
-- and specific needs. This is an example for HNSW.
-- Ensure you know the implications before adding specific index types.
-- CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING hnsw (embedding vector_l2_ops);
-- Or for older pgvector versions or different preferences:
-- CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

GRANT ALL PRIVILEGES ON TABLE documents TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON FUNCTION match_documents(vector, integer, jsonb) TO ${POSTGRES_USER};

-- Note: ${POSTGRES_USER} should be the user Flowise connects with.
-- If your init scripts don't support environment variable substitution directly like this,
-- you might need to hardcode the username or grant permissions in a separate step/script,
-- or ensure the default user has these permissions.
-- Often, the user defined by POSTGRES_USER in docker-compose is the owner and has rights.

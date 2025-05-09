-- postgres/init-flowise-schema.sql

-- Ensure the pgvector extension is enabled (idempotent)
-- This might be redundant if your init-pgvector.sql already does it,
-- but it's safe to include.
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the 'documents' table that Flowise expects
-- Note: id is TEXT for UUIDs, doc_metadata is JSONB
-- YOUR_VECTOR_SIZE should be replaced with the actual dimension, e.g., 1024
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  content TEXT,
  doc_metadata JSONB,
  embedding VECTOR(1024), -- Replace 1024 with your actual embedding dimension
  image_data BYTEA,       -- Added image_data column
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()) -- Added timestamp column
);

-- Create the 'match_documents' function that Flowise uses for searching
-- YOUR_VECTOR_SIZE should be replaced with the actual dimension, e.g., 1024
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1024), -- Replace 1024 with your actual embedding dimension
  match_count INT DEFAULT 5,
  filter JSONB DEFAULT '{}'
) RETURNS TABLE (
  id TEXT,
  content TEXT,
  doc_metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.doc_metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents AS d
  WHERE (filter IS NULL OR d.doc_metadata @> filter)
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

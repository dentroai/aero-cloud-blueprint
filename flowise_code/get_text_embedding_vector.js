const fetch = require('node-fetch');

async function getEmbeddingVector() {
  const url = "http://vllm_text_embedder:8000/v1/embeddings";

  const modelName = "vllm-text-embedding-model";

  // $question is the input variable provided by Flowise from "Input Variables"
  const payload = {
    input: $question, 
    model: modelName
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`HTTP error when calling embedding service! Status: ${response.status}, Body: ${errorBody}`);

      return JSON.stringify({ error: `Error fetching embedding: ${response.status} - ${errorBody}` });
    }

    const responseData = await response.json();

    // The OpenAI-compatible API returns embeddings in a structure like:
    // { "object": "list", "data": [ { "object": "embedding", "index": 0, "embedding": [...] } ], ... }
    if (responseData && responseData.data && responseData.data.length > 0 && responseData.data[0].embedding) {
      // Successfully retrieved the embedding vector.
      // Custom functions must return a string. We'll stringify the array.
      return JSON.stringify(responseData.data[0].embedding);
    } else {
      console.error("Embedding data not found in the expected format in API response:", responseData);
      return JSON.stringify({ error: "Error: Embedding format incorrect or embedding not found in API response." });
    }

  } catch (error) {
    console.error("Error within custom JavaScript function (getEmbeddingVector):", error);
    
    return JSON.stringify({ error: `Error in custom function execution: ${error.message}` });
  }
}

// Flowise will execute this. The return value (or the resolved value of the Promise) 
// from getEmbeddingVector() will be the output of this custom function node.
// Ensure the final returned value is a string.
return getEmbeddingVector();
const fetch = require('node-fetch');

async function generateImageEmbedding() {
  // --- Input from Flowise ---
  const inputText = $userQuestion;

  // --- Hardcoded vLLM Image Embedder Configuration ---
  const embedderHost = 'vllm_image_embedder'; // Service name from docker-compose.yaml
  const embedderPort = 8000; // Internal port vLLM listens on
  const embedderModelName = 'vllm-image-embedding-model';

  if (!inputText) {
    console.error("Input text ($userQuestion) is missing for image embedding generation.");
    return JSON.stringify({ error: "Missing input text for image embedding." });
  }

  const apiUrl = `http://${embedderHost}:${embedderPort}/v1/embeddings`;
  const payload = {
    input: inputText,
    model: embedderModelName
  };

  console.log("Sending payload to vLLM Image Embedder:", JSON.stringify(payload));

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`vLLM Image Embedder API error! Status: ${response.status}, Body: ${errorBody}`);
      return JSON.stringify({ error: `Image Embedder API request failed: ${response.status} - ${errorBody}` });
    }

    const responseData = await response.json();
    console.log("vLLM Image Embedder Response Data received.");

    // Standard OpenAI embedding response format
    if (responseData && responseData.data && responseData.data.length > 0 && responseData.data[0].embedding) {
      // Custom functions must return a string. Stringify the array.
      return JSON.stringify(responseData.data[0].embedding);
    } else {
      console.error("Image embedding data not found in the expected format:", responseData);
      return JSON.stringify({ error: "Image embedding format incorrect or embedding not found." });
    }

  } catch (error) {
    console.error("Error calling vLLM Image Embedder API:", error);
    return JSON.stringify({ error: `Error during Image Embedder API call: ${error.message}` });
  }
}

return generateImageEmbedding();
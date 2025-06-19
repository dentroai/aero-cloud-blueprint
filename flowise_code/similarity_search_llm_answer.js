const fetch = require('node-fetch');
const { Pool } = require('pg');

// Helper function (same as before)
function imageBytesToDataUri(imageBuffer, mimeType = 'image/jpeg') {
  if (!imageBuffer || imageBuffer.length === 0) return null;
  return `data:${mimeType};base64,${imageBuffer.toString('base64')}`;
}

async function searchContextAndGenerateMultimodalResponse() {
  // --- Inputs from Flowise ---
  const inputTextEmbeddingString = $textEmbeddingVectorString;
  const inputImageEmbeddingString = $imageEmbeddingVectorString;
  const question = $userQuestion;
  const conversationHistory = $conversationHistory;

  // --- Hardcoded PostgreSQL Configuration ---
  const dbHost = 'postgres';
  const dbPort = 5432;
  const dbUser = 'rag_user';
  const dbPassword = 'arnoldalsostartedingraz'; // <-- IMPORTANT: Replace!
  const dbName = 'rag_db';
  const textMatchCount = 5; // How many text chunks
  const imageMatchCount = 5; // How many images
  const dbFilterJsonString = '{}';

  // --- Hardcoded vLLM LLM (Multi-modal) Configuration ---
  const llmHost = 'vllm_llm';
  const llmPort = 8000;
  const llmModelName = 'vllm-llm-model'; // Example, use your actual model
  const maxOutputTokens = 5000;
  const llmTemp = 0.3;

  // Basic input validation
  if (!inputTextEmbeddingString || !inputImageEmbeddingString || !question) {
    console.error("Text embedding, image embedding, or user question is missing.");
    return JSON.stringify({ 
      error: "Missing required inputs for combined search and LLM call.",
      llm_answer: null,
      text_sources: [],
      image_sources: []
    });
  }
  if (dbPassword === 'YOUR_POSTGRES_PASSWORD') {
    console.error("CRITICAL: Default PostgreSQL password is still in the script. Please replace it!");
    return JSON.stringify({ 
      error: "PostgreSQL password not configured in the script.",
      llm_answer: null,
      text_sources: [],
      image_sources: []
    });
  }

  let textEmbeddingVector, imageEmbeddingVector;
  try {
    textEmbeddingVector = JSON.parse(inputTextEmbeddingString);
    imageEmbeddingVector = JSON.parse(inputImageEmbeddingString); // Parse new input
  } catch (e) {
    console.error("Failed to parse one or both embedding strings:", e);
    return JSON.stringify({ 
      error: `Invalid embedding vector format: ${e.message}`,
      llm_answer: null,
      text_sources: [],
      image_sources: []
    });
  }

  let dbFilter;
  try {
    dbFilter = JSON.parse(dbFilterJsonString);
  } catch (e) {
    console.error("Failed to parse dbFilterJsonString:", e);
    return JSON.stringify({ 
      error: `Invalid DB filter JSON format: ${e.message}`,
      llm_answer: null,
      text_sources: [],
      image_sources: []
    });
  }

  const pool = new Pool({ user: dbUser, host: dbHost, database: dbName, password: dbPassword, port: dbPort });
  const client = await pool.connect();

  let retrievedTextContext = "Keine spezifischen Textinformationen gefunden.";
  const llmImageInputs = []; // To store base64 image URIs for the LLM
  const textSources = []; // To store text sources for response
  const imageSources = []; // To store image sources for response

  try {
    // --- 1a. Text Similarity Search ---
    console.log("Step 1a: Performing TEXT similarity search...");
    const queryTextEmbeddingSql = JSON.stringify(textEmbeddingVector);
    const sqlTextQuery = `SELECT "pageContent" FROM match_documents($1, $2, $3);`; // Only need pageContent
    const textRes = await client.query(sqlTextQuery, [queryTextEmbeddingSql, textMatchCount, dbFilter]);
    
    if (textRes.rows && textRes.rows.length > 0) {
      retrievedTextContext = textRes.rows.map(row => {
          const contentKey = Object.keys(row).find(key => key.toLowerCase() === 'pagecontent');
          const content = contentKey ? row[contentKey] : '';
          
          // Add to text sources
          if (content) {
            textSources.push({
              content: content,
              type: 'text'
            });
          }
          
          return content;
      }).join("\n\n---\n\n");
      console.log(`Retrieved ${textRes.rows.length} text chunks.`);
    } else {
      console.log("No text chunks retrieved.");
    }

    // --- 1b. Image Similarity Search using the new image embedding ---
    console.log("Step 1b: Performing IMAGE similarity search...");
    const queryImageEmbeddingSql = JSON.stringify(imageEmbeddingVector);
    // IMPORTANT: Use the new SQL function `match_images_by_embedding`
    const sqlImageQuery = `SELECT id, "image_data" FROM match_images_by_embedding($1, $2, $3);`;
    const imageRes = await client.query(sqlImageQuery, [queryImageEmbeddingSql, imageMatchCount, dbFilter]);

    if (imageRes.rows && imageRes.rows.length > 0) {
      imageRes.rows.forEach(row => {
        if (row.image_data && row.image_data.length > 0) {
          const dataUri = imageBytesToDataUri(row.image_data, 'image/jpeg'); // Assuming JPEG
          if (dataUri) {
            llmImageInputs.push({ type: "image_url", image_url: { "url": dataUri } });
            
            // Add to image sources
            imageSources.push({
              id: row.id,
              data_uri: dataUri,
              type: 'image'
            });
            
            console.log(`Processed image data for ID ${row.id} into data URI.`);
          }
        }
      });
      console.log(`Retrieved ${imageRes.rows.length} rows from image search. Processed ${llmImageInputs.length} images for LLM.`);
    } else {
      console.log("No images retrieved from image similarity search.");
    }

  } catch (err) {
    console.error('Error during PostgreSQL search operations:', err.stack);
    retrievedTextContext = retrievedTextContext === "Keine spezifischen Textinformationen gefunden." ? `Fehler bei Textsuche: ${err.message}` : retrievedTextContext;
    if (llmImageInputs.length === 0) {
        console.warn("Proceeding without images due to DB error or no results.");
    }
  } finally {
    client.release();
    await pool.end(); // Important to close the pool after all DB ops
    console.log("PostgreSQL client released and pool ended.");
  }

  // --- 2. Call vLLM Multi-modal LLM ---
  console.log("Step 2: Calling vLLM Multi-modal LLM...");
  const combinedTextPromptForLLM = `Du bist ein hilfsbereiter Assistent. Beantworte die Frage des Benutzers basierend auf dem folgenden Textkontext und den bereitgestellten Bildern. Wenn der Kontext nicht ausreicht, teile dies mit.

Textkontext:
---
${retrievedTextContext}
---

Bestehende Konversation:
---
${conversationHistory}
---

Frage des Benutzers: ${question}`;

  const llmApiContent = [{ type: "text", text: combinedTextPromptForLLM }];
  llmImageInputs.forEach(imgContent => llmApiContent.push(imgContent)); // Add processed images

  const llmApiUrl = `http://${llmHost}:${llmPort}/v1/chat/completions`;
  const llmPayload = {
    model: llmModelName,
    messages: [{ role: "user", content: llmApiContent }],
    temperature: llmTemp,
    max_tokens: maxOutputTokens
  };

  console.log("Sending payload to LLM (first 1000 chars):", JSON.stringify(llmPayload, null, 2).substring(0, 1000) + "...");

  try {
    const llmResponse = await fetch(llmApiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(llmPayload)
    });

    if (!llmResponse.ok) {
      const errorBody = await llmResponse.text();
      console.error(`LLM API error! Status: ${llmResponse.status}, Body: ${errorBody}`);
      return JSON.stringify({ 
        error: `LLM API request failed: ${llmResponse.status} - ${errorBody}`,
        llm_answer: null,
        text_sources: textSources,
        image_sources: imageSources
      });
    }
    
    const responseData = await llmResponse.json();
    if (responseData && responseData.choices && responseData.choices.length > 0 && responseData.choices[0].message && responseData.choices[0].message.content) {
      // Return structured JSON response
      return JSON.stringify({
        llm_answer: responseData.choices[0].message.content,
        text_sources: textSources,
        image_sources: imageSources
      });
    } else {
      console.error("LLM response format incorrect or content missing:", responseData);
      return JSON.stringify({ 
        error: "LLM response format incorrect or content missing.",
        llm_answer: null,
        text_sources: textSources,
        image_sources: imageSources
      });
    }
  } catch (error) {
    console.error("Error calling LLM API:", error);
    return JSON.stringify({ 
      error: `Error during LLM API call: ${error.message}`,
      llm_answer: null,
      text_sources: textSources,
      image_sources: imageSources
    });
  }
}

return searchContextAndGenerateMultimodalResponse();
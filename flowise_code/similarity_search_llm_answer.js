const fetch = require('node-fetch');
const { Pool } = require('pg');

// --- Configuration Constants ---
const ASSUMED_IMAGE_MIME_TYPE = 'image/jpeg'; // Default MIME type for images from DB

// Helper function to convert raw image bytes (from DB) to base64 data URI
function imageBytesToDataUri(imageBuffer, targetMimeType) {
  if (!imageBuffer || imageBuffer.length === 0) return null;
  // Image resizing/reformatting should ideally happen in the ETL pipeline *before* DB storage.
  // This function converts the provided imageBuffer (as-is from DB) to a base64 data URI.
  const base64Encoded = imageBuffer.toString('base64');
  return `data:${targetMimeType};base64,${base64Encoded}`;
}

async function searchContextAndGenerateMultimodalResponse() {
  // --- Inputs from Flowise ---
  const inputTextEmbeddingString = $textEmbeddingVectorString;
  const inputImageEmbeddingString = $imageEmbeddingVectorString;
  const question = $userQuestion;

  // --- Hardcoded PostgreSQL Configuration ---
  const dbHost = 'postgres';
  const dbPort = 5432;
  const dbUser = 'rag_user';
  const dbPassword = 'your_strong_postgres_password'; 
  const dbName = 'rag_db';
  const textMatchCount = 5; 
  const imageMatchCount = 5; 
  const dbFilterJsonString = '{}';

  // --- Hardcoded vLLM LLM (Multi-modal) Configuration ---
  const llmHost = 'vllm_llm';
  const llmPort = 8000;
  const llmModelName = 'vllm-llm-model'; 
  const maxOutputTokens = 5000;
  const llmTemp = 0.3;

  // --- Variables to store results ---
  let llmAnswer = "Das LLM konnte keine Antwort generieren."; // Default LLM answer
  const retrievedTextSources = [];
  const retrievedImageSources = []; // Will store {id, metadata, image_base64_data_uri}

  // Basic input validation
  if (!inputTextEmbeddingString || !inputImageEmbeddingString || !question) {
    const errorMsg = "Text embedding, image embedding, or user question is missing.";
    console.error(errorMsg);
    return JSON.stringify({ llm_answer: `Fehler: ${errorMsg}`, text_sources: [], image_sources: [] });
  }

  let textEmbeddingVector, imageEmbeddingVector;
  try {
    textEmbeddingVector = JSON.parse(inputTextEmbeddingString);
    imageEmbeddingVector = JSON.parse(inputImageEmbeddingString); 
  } catch (e) {
    const errorMsg = `Invalid embedding vector format: ${e.message}`;
    console.error("Failed to parse one or both embedding strings:", e);
    return JSON.stringify({ llm_answer: `Fehler: ${errorMsg}`, text_sources: [], image_sources: [] });
  }

  let dbFilter;
  try {
    dbFilter = JSON.parse(dbFilterJsonString);
  } catch (e) {
    const errorMsg = `Invalid DB filter JSON format: ${e.message}`;
    console.error("Failed to parse dbFilterJsonString:", e);
    return JSON.stringify({ llm_answer: `Fehler: ${errorMsg}`, text_sources: [], image_sources: [] });
  }

  const pool = new Pool({ user: dbUser, host: dbHost, database: dbName, password: dbPassword, port: dbPort });
  const client = await pool.connect();

  let retrievedTextContextForLLM = "Keine spezifischen Textinformationen gefunden.";
  const llmImageInputsForLLM = []; // To store {type: "image_url", ...} for the LLM API

  try {
    // --- 1a. Text Similarity Search ---
    console.log("Step 1a: Performing TEXT similarity search...");
    const queryTextEmbeddingSql = JSON.stringify(textEmbeddingVector);
    // Fetch pageContent and metadata for text sources
    const sqlTextQuery = `SELECT "pageContent", metadata FROM match_documents($1, $2, $3);`; 
    const textRes = await client.query(sqlTextQuery, [queryTextEmbeddingSql, textMatchCount, dbFilter]);
    
    if (textRes.rows && textRes.rows.length > 0) {
      const tempTextContexts = [];
      textRes.rows.forEach(row => {
        const pageContentKey = Object.keys(row).find(key => key.toLowerCase() === 'pagecontent');
        const content = pageContentKey ? row[pageContentKey] : '';
        tempTextContexts.push(content);
        retrievedTextSources.push({
          pageContent: content,
          metadata: row.metadata || {} // Ensure metadata is at least an empty object
        });
      });
      retrievedTextContextForLLM = tempTextContexts.join("\n\n---\n\n");
      console.log(`Retrieved ${retrievedTextSources.length} text chunks for sources.`);
    } else {
      console.log("No text chunks retrieved.");
    }

    // --- 1b. Image Similarity Search ---
    console.log("Step 1b: Performing IMAGE similarity search...");
    const queryImageEmbeddingSql = JSON.stringify(imageEmbeddingVector);
    // Fetch id, metadata, and image_data for image sources
    const sqlImageQuery = `SELECT id, metadata, "image_data" FROM match_images_by_embedding($1, $2, $3);`;
    const imageRes = await client.query(sqlImageQuery, [queryImageEmbeddingSql, imageMatchCount, dbFilter]);

    if (imageRes.rows && imageRes.rows.length > 0) {
      imageRes.rows.forEach(row => {
        if (row.image_data && row.image_data.length > 0) {
          const dataUri = imageBytesToDataUri(row.image_data, ASSUMED_IMAGE_MIME_TYPE); 
          if (dataUri) {
            // For LLM API payload
            llmImageInputsForLLM.push({ type: "image_url", image_url: { "url": dataUri } });
            // For final output sources
            retrievedImageSources.push({
              id: row.id,
              metadata: row.metadata || {}, // Ensure metadata is at least an empty object
              image_base64_data_uri: dataUri
            });
            console.log(`Processed image data for ID ${row.id} for LLM and sources.`);
          }
        }
      });
      console.log(`Retrieved ${imageRes.rows.length} rows from image search. Prepared ${llmImageInputsForLLM.length} images for LLM and ${retrievedImageSources.length} for sources.`);
    } else {
      console.log("No images retrieved from image similarity search.");
    }

  } catch (err) {
    console.error('Error during PostgreSQL search operations:', err.stack);
    // Update default context for LLM if it's still the initial default
    if (retrievedTextContextForLLM === "Keine spezifischen Textinformationen gefunden.") {
        retrievedTextContextForLLM = `Fehler bei Textsuche: ${err.message}`;
    }
    // Note: llmImageInputsForLLM would be empty if image search failed.
  } finally {
    client.release();
    await pool.end(); 
    console.log("PostgreSQL client released and pool ended.");
  }

  // --- 2. Call vLLM Multi-modal LLM ---
  console.log("Step 2: Calling vLLM Multi-modal LLM...");
  const combinedTextPromptForLLM = `Du bist ein hilfsbereiter Assistent. Beantworte die Frage des Benutzers basierend auf dem folgenden Textkontext und den bereitgestellten Bildern. Wenn der Kontext nicht ausreicht, teile dies mit.

Textkontext:
---
${retrievedTextContextForLLM}
---

Frage des Benutzers: ${question}`;

  const llmApiContent = [{ type: "text", text: combinedTextPromptForLLM }];
  llmImageInputsForLLM.forEach(imgContent => llmApiContent.push(imgContent)); 

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
      llmAnswer = `LLM API Anfrage fehlgeschlagen: ${response.status} - ${errorBody}`;
    } else {
      const responseData = await llmResponse.json();
      if (responseData && responseData.choices && responseData.choices.length > 0 && responseData.choices[0].message && responseData.choices[0].message.content) {
        llmAnswer = responseData.choices[0].message.content;
      } else {
        console.error("LLM response format incorrect or content missing:", responseData);
        llmAnswer = "Das LLM gab eine Antwort in einem unerwarteten Format zurück oder der Inhalt fehlt.";
      }
    }
  } catch (error) {
    console.error("Error calling LLM API:", error);
    llmAnswer = `Fehler bei der LLM API Anfrage: ${error.message}`;
  }

  // --- 3. Construct and return the final JSON object ---
  const finalOutput = {
    llm_answer: llmAnswer,
    text_sources: retrievedTextSources,
    image_sources: retrievedImageSources
  };

  console.log("Final output being returned (llm_answer and counts):", 
    JSON.stringify({llm_answer: finalOutput.llm_answer, text_sources_count: finalOutput.text_sources.length, image_sources_count: finalOutput.image_sources.length })
  );

  return JSON.stringify(finalOutput);
}

return searchContextAndGenerateMultimodalResponse();
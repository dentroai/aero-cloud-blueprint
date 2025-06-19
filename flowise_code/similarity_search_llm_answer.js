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
  const dbPassword = 'arnoldalsostartedingraz';
  const dbName = 'rag_db';
  const textMatchCount = 5; // How many text chunks
  const imageMatchCount = 5; // How many images
  const dbFilterJsonString = '{}';

  // --- Hardcoded vLLM LLM (Multi-modal) Configuration ---
  const llmHost = 'vllm_llm';
  const llmPort = 8000;
  const llmModelName = 'vllm-llm-model';
  const maxOutputTokens = 5000;
  const llmTemp = 0.3;

  // Basic input validation
  if (!inputTextEmbeddingString || !inputImageEmbeddingString || !question) {
    console.error("Text embedding, image embedding, or user question is missing.");
    // Return the response in the expected format
    return {
      text: JSON.stringify({ 
        error: "Missing required inputs for combined search and LLM call.",
        llm_answer: "Missing required inputs for combined search and LLM call.",
        text_sources: [],
        image_sources: []
      })
    };
  }

  let textEmbeddingVector, imageEmbeddingVector;
  try {
    textEmbeddingVector = JSON.parse(inputTextEmbeddingString);
    imageEmbeddingVector = JSON.parse(inputImageEmbeddingString);
  } catch (e) {
    console.error("Failed to parse one or both embedding strings:", e);
    return {
      text: JSON.stringify({ 
        error: `Invalid embedding vector format: ${e.message}`,
        llm_answer: `Invalid embedding vector format: ${e.message}`,
        text_sources: [],
        image_sources: []
      })
    };
  }

  let dbFilter;
  try {
    dbFilter = JSON.parse(dbFilterJsonString);
  } catch (e) {
    console.error("Failed to parse dbFilterJsonString:", e);
    return {
      text: JSON.stringify({ 
        error: `Invalid DB filter JSON format: ${e.message}`,
        llm_answer: `Invalid DB filter JSON format: ${e.message}`,
        text_sources: [],
        image_sources: []
      })
    };
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
    const sqlTextQuery = `SELECT "pageContent", metadata FROM match_documents($1, $2, $3);`;
    const textRes = await client.query(sqlTextQuery, [queryTextEmbeddingSql, textMatchCount, dbFilter]);
    
    if (textRes.rows && textRes.rows.length > 0) {
      const textChunks = [];
      textRes.rows.forEach((row, index) => {
        const contentKey = Object.keys(row).find(key => key.toLowerCase() === 'pagecontent');
        const content = contentKey ? row[contentKey] : '';
        
        if (content) {
          textChunks.push(content);
          
          // Create properly formatted text source
          textSources.push({
            id: `text-source-${index + 1}`,
            title: row.metadata?.title || row.metadata?.source || `Textquelle ${index + 1}`,
            content: content,
            pageContent: content, // Include both for compatibility
            page: row.metadata?.page_number?.toString() || row.metadata?.page || undefined,
            lastUpdated: row.metadata?.lastUpdated || undefined,
            url: row.metadata?.url || undefined,
            metadata: row.metadata || {}
          });
        }
      });
      
      retrievedTextContext = textChunks.join("\n\n---\n\n");
      console.log(`Retrieved ${textRes.rows.length} text chunks.`);
    } else {
      console.log("No text chunks retrieved.");
    }

    // --- 1b. Image Similarity Search using the new image embedding ---
    console.log("Step 1b: Performing IMAGE similarity search...");
    const queryImageEmbeddingSql = JSON.stringify(imageEmbeddingVector);
    const sqlImageQuery = `SELECT id, "image_data", metadata FROM match_images_by_embedding($1, $2, $3);`;
    const imageRes = await client.query(sqlImageQuery, [queryImageEmbeddingSql, imageMatchCount, dbFilter]);

    if (imageRes.rows && imageRes.rows.length > 0) {
      imageRes.rows.forEach((row, index) => {
        if (row.image_data && row.image_data.length > 0) {
          const dataUri = imageBytesToDataUri(row.image_data, 'image/jpeg');
          if (dataUri) {
            llmImageInputs.push({ type: "image_url", image_url: { "url": dataUri } });
            
            // Create properly formatted image source
            imageSources.push({
              id: row.id || `image-source-${index + 1}`,
              base64: dataUri,
              image_base64_data_uri: dataUri, // Include this field for compatibility
              metadata: {
                file_name: row.metadata?.file_name || `Bild ${index + 1}`,
                file_path: row.metadata?.file_path || undefined,
                page_number: row.metadata?.page_number || undefined,
                original_timestamp: row.metadata?.original_timestamp || undefined,
                ...row.metadata
              }
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
${retrievedTextContext}
---

Bestehende Konversation:
---
${conversationHistory}
---

Frage des Benutzers: ${question}`;

  const llmApiContent = [{ type: "text", text: combinedTextPromptForLLM }];
  llmImageInputs.forEach(imgContent => llmApiContent.push(imgContent));

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
      return {
        text: JSON.stringify({ 
          error: `LLM API request failed: ${llmResponse.status} - ${errorBody}`,
          llm_answer: `LLM API request failed: ${llmResponse.status}`,
          text_sources: textSources,
          image_sources: imageSources
        })
      };
    }
    
    const responseData = await llmResponse.json();
    if (responseData && responseData.choices && responseData.choices.length > 0 && responseData.choices[0].message && responseData.choices[0].message.content) {
      // The chat app expects the response in the 'text' field as a JSON string
      return {
        llm_answer: responseData.choices[0].message.content,
        text_sources: textSources,
        image_sources: imageSources
      };
    } else {
      console.error("LLM response format incorrect or content missing:", responseData);
      return {
        text: JSON.stringify({ 
          error: "LLM response format incorrect or content missing.",
          llm_answer: "LLM response format incorrect or content missing.",
          text_sources: textSources,
          image_sources: imageSources
        })
      };
    }
  } catch (error) {
    console.error("Error calling LLM API:", error);
    return {
      text: JSON.stringify({ 
        error: `Error during LLM API call: ${error.message}`,
        llm_answer: `Error during LLM API call: ${error.message}`,
        text_sources: textSources,
        image_sources: imageSources
      })
    };
  }
}

return searchContextAndGenerateMultimodalResponse();
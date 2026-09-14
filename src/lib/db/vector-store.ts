import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { getCritiqueById } from "./database";
import path from "path";
import fs from "fs";
import { Document } from "langchain/document";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { SimpleVectorStore } from "./simple-vector-store";

// Directory for storing the vector store data
const VECTOR_DIR = path.join(process.cwd(), 'data', 'vectors');
const VECTOR_DATA_FILE = path.join(VECTOR_DIR, 'vector-data.json');


let vectorStore: SimpleVectorStore | null = null;
let initialization: Promise<boolean> | null = null;

// Publish the store only after all saved documents have been embedded successfully.
// Failed initialization leaves no partial index and can be retried.
export async function initVectorStore(): Promise<boolean> {
  if (vectorStore) return true;
  if (initialization) return initialization;
  initialization = (async () => {
    fs.mkdirSync(VECTOR_DIR, { recursive: true });
    const embeddings = new OllamaEmbeddings({
      model: 'llama3',
      baseUrl: 'http://localhost:11434',
    });
    const candidate = new SimpleVectorStore(embeddings);
    if (fs.existsSync(VECTOR_DATA_FILE)) {
      const savedData = JSON.parse(fs.readFileSync(VECTOR_DATA_FILE, 'utf-8'));
      if (!Array.isArray(savedData)) throw new Error('Invalid vector store file: expected an array.');
      const documents = savedData
        // Discard the historical placeholder; it is not a retrieved code example.
        .filter(item => item?.content?.metadata?.id !== 'placeholder')
        .map(item => {
          if (typeof item?.content?.pageContent !== 'string') {
            throw new Error('Invalid saved vector document.');
          }
          return new Document({
            pageContent: item.content.pageContent,
            metadata: item.content.metadata || {},
          });
        });
      // Re-embed stored text instead of trusting historical synthetic vectors.
      await candidate.addDocuments(documents);
    }
    vectorStore = candidate;
    return true;
  })();
  try {
    return await initialization;
  } finally {
    initialization = null;
  }
}

// Split code into chunks for embedding
export async function splitCodeIntoChunks(code: string, language: string) {
  // Use a text splitter appropriate for code
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });
  
  // Split the code into chunks
  const docs = await splitter.createDocuments(
    [code],
    [{ language, type: "code" }]
  );
  
  return docs;
}

// Save vector store data to disk
async function saveVectorStore() {
  try {
    if (!vectorStore) return;
    
    // Get all vectors
    const vectors = vectorStore.allVectors;
    
    // Save to file
    fs.writeFileSync(
      VECTOR_DATA_FILE, 
      JSON.stringify(vectors, null, 2)
    );
    
    console.log(`Saved ${vectors.length} documents to vector store file`);
    return true;
  } catch (error) {
    console.error("Error saving vector store:", error);
    return false;
  }
}

// Add a code sample to the vector store
export async function addToVectorStore(code: string, critiqueId: string, language: string) {
  try {
    // Ensure vector store is initialized
    if (!vectorStore) {
      await initVectorStore();
    }
    
    // Split the code into chunks
    const chunks = await splitCodeIntoChunks(code, language);
    
    // Add metadata to each chunk
    const docsWithMetadata = chunks.map(chunk => {
      return new Document({
        pageContent: chunk.pageContent,
        metadata: {
          ...chunk.metadata,
          critiqueId,
          timestamp: new Date().toISOString(),
        }
      });
    });
    
    // Add to vector store
    await vectorStore!.addDocuments(docsWithMetadata);
    
    // Save the updated vector store
    await saveVectorStore();
    
    return true;
  } catch (error) {
    console.error("Error adding to vector store:", error);
    throw error;
  }
}

// Find similar code examples
export async function findSimilarCode(code: string, limit = 5) {
  try {
    // Ensure vector store is initialized
    if (!vectorStore) {
      await initVectorStore();
    }
    
    // Get embeddings for the query code
    const results = await vectorStore!.similaritySearch(code, limit);
    
    // Augment results with critique data if needed
    const enhancedResults = await Promise.all(
      results.map(async (result) => {
        if (result.metadata.critiqueId && result.metadata.critiqueId !== "placeholder") {
          try {
            const critique = await getCritiqueById(result.metadata.critiqueId);
            if (critique) {
              return {
                ...result,
                metadata: {
                  ...result.metadata,
                  critique: {
                    summary: critique.summary,
                    language: critique.language,
                    issueCount: critique.issues.length,
                  },
                },
              };
            }
          } catch (error) {
            console.warn(`Error fetching critique ${result.metadata.critiqueId}:`, error);
          }
        }
        return result;
      })
    );
    
    return enhancedResults;
  } catch (error) {
    console.error("Error searching vector store:", error);
    
    // Retrieval failure must reach the caller, rather than look like no matches.
    throw error;
  }
} 

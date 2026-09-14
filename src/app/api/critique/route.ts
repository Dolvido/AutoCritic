import { NextResponse } from "next/server";
import { critiqueCode } from "@/lib/llm/critic";
import { initializeDatabase } from "@/lib/db/database";
import { initVectorStore } from "@/lib/db/vector-store";
import { getOptimizedExamples, scheduleMetaAgent } from "@/lib/llm/meta-agent";

// Initialize database and vector store
let initialized = false;
const initialize = async () => {
  if (!initialized) {
    try {
      await initializeDatabase();
      await initVectorStore();
      
      // Schedule the meta-agent to run periodically
      scheduleMetaAgent(24); // Run every 24 hours
      
      initialized = true;
      console.log("Database, vector store, and meta-agent initialized successfully");
    } catch (error) {
      console.error("Failed to initialize:", error);
      throw error;
    }
  }
};

export async function POST(request: Request) {
  try {
    // Initialize the database and vector store if not already done
    await initialize();
    
    const body = await request.json();
    const { code, language } = body;

    if (!code || !language) {
      return NextResponse.json(
        { error: "Code and language are required" },
        { status: 400 }
      );
    }

    // Get optimized examples from meta-agent
    const optimizedExamples = await getOptimizedExamples(2);
    
    // Process the code with our LLM service
    const critique = await critiqueCode(code, language, {
      temperature: 0.3,
      useSimilarExamples: true,
      maxExamples: 2,
      // Only use optimized examples if we have some
      customExamples: optimizedExamples.length > 0 ? optimizedExamples : undefined
    });

    return NextResponse.json({ critique });
  } catch (error) {
    console.error("Error processing critique:", error);
    return NextResponse.json(
      { error: "Critique unavailable. Check the local Ollama service, models, and application logs." },
      { status: 500 }
    );
  }
} 

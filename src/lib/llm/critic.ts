import { v4 as uuidv4 } from 'uuid';
import type { CritiqueResult } from '@/components/CritiqueResults';
import { findSimilarCode } from '../db/vector-store';
import { saveCritique } from '../db/database';
import { addToVectorStore } from '../db/vector-store';
import { Ollama } from '@langchain/community/llms/ollama';
import { PromptTemplate } from '@langchain/core/prompts';

interface CritiqueOptions {
  temperature?: number;
  maxTokens?: number;
  useSimilarExamples?: boolean;
  maxExamples?: number;
  model?: string;
  customExamples?: Array<{
    code: string;
    critique: {
      summary: string;
      issues: Array<{
        title: string;
        description: string;
        fixSuggestion: string;
        severity: string;
      }>;
    }
  }>;
}

// System prompt template
const systemPrompt = `You are an expert code reviewer named Auto-Critic. Your task is to analyze the given code and provide a structured critique.
Focus on:
1. Code quality
2. Best practices
3. Potential bugs or edge cases
4. Performance issues
5. Readability and maintainability

Format your response in JSON with the following structure:
{
  "summary": "Brief overview of the code and major findings",
  "issues": [
    {
      "title": "Brief title of the issue",
      "description": "Detailed explanation of the problem",
      "fixSuggestion": "Code example showing how to fix the issue",
      "severity": "high|medium|low"
    }
  ]
}
`;

// Default few-shot examples for the LLM
const getDefaultFewShotExamples = () => {
  return [
    {
      code: `function calculateTotal(items) {
  let total = 0;
  for (let i = 0; i < items.length; i++) {
    total = total + items[i].price;
  }
  return total;
}`,
      critique: {
        summary: "This function calculates the total price of items in an array but has some issues with error handling and could be more concise.",
        issues: [
          {
            title: "No input validation",
            description: "The function doesn't check if 'items' is a valid array or if items have a 'price' property.",
            fixSuggestion: "function calculateTotal(items) {\n  if (!Array.isArray(items)) return 0;\n  \n  let total = 0;\n  for (let i = 0; i < items.length; i++) {\n    if (items[i] && typeof items[i].price === 'number') {\n      total += items[i].price;\n    }\n  }\n  return total;\n}",
            severity: "high"
          },
          {
            title: "Could use array method",
            description: "The function uses a for loop when it could use the more concise reduce() method.",
            fixSuggestion: "function calculateTotal(items) {\n  if (!Array.isArray(items)) return 0;\n  \n  return items.reduce((total, item) => {\n    return total + (item && typeof item.price === 'number' ? item.price : 0);\n  }, 0);\n}",
            severity: "medium"
          }
        ]
      }
    }
  ];
};

// Get few-shot examples, either custom or default
const getFewShotExamples = (options: CritiqueOptions) => {
  // Use custom examples if provided
  if (options.customExamples && options.customExamples.length > 0) {
    return options.customExamples;
  }
  
  // Otherwise use the default examples
  return getDefaultFewShotExamples();
};

// Function to build the prompt
const buildPrompt = async (code: string, language: string, options: CritiqueOptions) => {
  const examples = getFewShotExamples(options);
  let exampleText = '';
  
  // Add few-shot examples to the prompt
  examples.forEach((example, index) => {
    exampleText += `Example ${index + 1}:\nCode:\n\`\`\`\n${example.code}\n\`\`\`\n\nCritique:\n\`\`\`json\n${JSON.stringify(example.critique, null, 2)}\n\`\`\`\n\n`;
  });
  
  // Get similar examples if requested
  if (options.useSimilarExamples) {
    const similarExamples = await findSimilarCode(code, options.maxExamples || 2);
    if (similarExamples.length > 0) {
      exampleText += 'Similar examples from history:\n\n';
      
      for (const example of similarExamples) {
        if (example.metadata?.critiqueId) {
          exampleText += `Code:\n\`\`\`\n${example.pageContent.substring(0, 300)}${example.pageContent.length > 300 ? '...' : ''}\n\`\`\`\n\n`;
          
          // Add critique information if available
          if (example.metadata?.critique) {
            exampleText += `Critique Summary: ${example.metadata.critique.summary}\n\n`;
          }
        }
      }
    }
  }
  
  // Build the full prompt
  const promptText = `${systemPrompt}

${exampleText}

Now critique the following code in ${language}:

\`\`\`
${code}
\`\`\`

Remember to format your response as a valid JSON object with "summary" and "issues" fields.`;

  return promptText;
};

// Real LLM implementation using Ollama
const generateCritique = async (code: string, language: string, options: CritiqueOptions) => {
  try {
    // Create Ollama client
    const model = options.model || 'codellama';
    const temperature = options.temperature || 0.3;
    
    const ollama = new Ollama({
      baseUrl: 'http://localhost:11434',
      model: model,
      temperature: temperature,
    });
    
    // Build the prompt
    const prompt = await buildPrompt(code, language, options);
    
    // Call the model
    const response = await ollama.call(prompt);
    
    // Parse the JSON response
    try {
      // Extract JSON part if the model added any preamble text
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      const jsonStr = jsonMatch ? jsonMatch[0] : response;
      
      const critiqueResponse = JSON.parse(jsonStr);
      
      // Add UUIDs to issues
      const issues = critiqueResponse.issues.map((issue: any) => ({
        ...issue,
        id: uuidv4()
      }));
      
      return {
        summary: critiqueResponse.summary,
        issues: issues
      };
    } catch (parseError) {
      console.error("Failed to parse LLM response as JSON:", parseError);
      console.log("Raw response:", response);
      throw new Error("LLM response format error");
    }
  } catch (error) {
    console.error("Error calling Ollama:", error);
    throw error;
  }
};

// Main function to critique code
export async function critiqueCode(
  code: string, 
  language: string, 
  options: CritiqueOptions = {}
): Promise<CritiqueResult> {
  try {
    // A failed model or retrieval request is an error, not a sample critique.
    const critiqueResponse = await generateCritique(code, language, options);
    
    // Generate a unique ID for this critique
    const id = uuidv4();
    
    // Create the final critique result
    const critiqueResult: CritiqueResult = {
      id,
      summary: critiqueResponse.summary,
      language,
      issues: critiqueResponse.issues,
      timestamp: new Date().toISOString()
    };
    
    // Save the critique to the database
    try {
      await saveCritique(critiqueResult, code);
      
      // Also add to vector store for future reference
      await addToVectorStore(code, id, language);
    } catch (dbError) {
      console.error("Error saving critique to database:", dbError);
      // Continue anyway - the critique is still useful even if it's not saved
    }
    
    return critiqueResult;
  } catch (error) {
    console.error("Error generating critique:", error);
    throw error;
  }
} 

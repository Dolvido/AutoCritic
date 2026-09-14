/**
 * A small exact cosine index. Embedding failures are errors, never synthetic data.
 * Persistence and the Ollama adapter live in vector-store.ts.
 */
export interface StoredDocument {
  pageContent: string;
  metadata: Record<string, any>;
}

export interface EmbeddingProvider {
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(text: string): Promise<number[]>;
}

type MemoryVector = { content: StoredDocument; embedding: number[] };

export class SimpleVectorStore {
  private vectors: MemoryVector[] = [];

  private embeddings: EmbeddingProvider;

  constructor(embeddings: EmbeddingProvider) {
    this.embeddings = embeddings;
  }

  private validate(vector: number[], expectedDimension?: number): void {
    if (!Array.isArray(vector) || vector.length === 0 ||
        vector.some(value => !Number.isFinite(value))) {
      throw new Error("Embedding must contain finite numeric values.");
    }
    if (expectedDimension !== undefined && vector.length !== expectedDimension) {
      throw new Error("Embedding dimension changed; use one model for the index.");
    }
    const norm = Math.hypot(...vector);
    if (!Number.isFinite(norm) || norm === 0) {
      throw new Error("Embedding must have a finite, non-zero norm.");
    }
  }

  async addDocuments(documents: StoredDocument[]): Promise<void> {
    if (documents.length === 0) return;
    let embeddings: number[][];
    try {
      embeddings = await this.embeddings.embedDocuments(documents.map(doc => doc.pageContent));
    } catch {
      throw new Error("Document embeddings unavailable. Check Ollama and the configured embedding model.");
    }
    if (!Array.isArray(embeddings) || embeddings.length !== documents.length) {
      throw new Error("Embedding count does not match the document count.");
    }
    const dimension = this.vectors[0]?.embedding.length ?? embeddings[0]?.length;
    // Validate the entire batch before mutating the index.
    embeddings.forEach(vector => this.validate(vector, dimension));
    this.vectors.push(...documents.map((content, index) => ({
      content,
      embedding: [...embeddings[index]]
    })));
  }

  async similaritySearch(query: string, k = 5): Promise<StoredDocument[]> {
    if (!Number.isInteger(k) || k < 0) throw new Error("Result limit must be a non-negative integer.");
    if (k === 0 || this.vectors.length === 0) return [];
    let queryEmbedding: number[];
    try {
      queryEmbedding = await this.embeddings.embedQuery(query);
    } catch {
      throw new Error("Query embedding unavailable. Check Ollama and the configured embedding model.");
    }
    this.validate(queryEmbedding, this.vectors[0].embedding.length);
    return this.vectors
      .map(vector => ({
        content: vector.content,
        score: this.cosineSimilarity(queryEmbedding, vector.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(result => result.content);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const normA = Math.hypot(...a);
    const normB = Math.hypot(...b);
    // Normalize first to avoid overflow in products of large coordinates.
    return a.reduce((sum, value, index) => sum + (value / normA) * (b[index] / normB), 0);
  }

  get allVectors(): MemoryVector[] {
    return this.vectors;
  }
}


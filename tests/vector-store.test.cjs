const assert = require('node:assert/strict');
const { test, before } = require('node:test');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

let SimpleVectorStore;
before(async () => {
  ({ SimpleVectorStore } = await import(pathToFileURL(
    path.resolve(__dirname, '../src/lib/db/simple-vector-store.ts')
  ).href));
});

const doc = pageContent => ({ pageContent, metadata: {} });
const provider = (vectors, query = [1, 0]) => ({
  embedDocuments: async () => vectors,
  embedQuery: async () => query
});

test('ranks real embeddings by cosine similarity and applies the result limit', async () => {
  const store = new SimpleVectorStore(provider([[0, 1], [1, 0], [-1, 0]]));
  await store.addDocuments([doc('perpendicular'), doc('aligned'), doc('opposite')]);
  assert.deepEqual((await store.similaritySearch('query', 2)).map(d => d.pageContent),
    ['aligned', 'perpendicular']);
});

test('a document embedding outage rejects without adding synthetic entries', async () => {
  let fail = false;
  const store = new SimpleVectorStore({
    embedDocuments: async () => {
      if (fail) throw new Error('Ollama offline');
      return [[1, 0]];
    },
    embedQuery: async () => [1, 0]
  });
  await store.addDocuments([doc('retained')]);
  fail = true;
  await assert.rejects(store.addDocuments([doc('must not appear')]), /Document embeddings unavailable/);
  assert.deepEqual((await store.similaritySearch('query')).map(d => d.pageContent), ['retained']);
});

test('a query embedding outage rejects rather than returning arbitrary matches', async () => {
  const store = new SimpleVectorStore({
    embedDocuments: async () => [[1, 0]],
    embedQuery: async () => { throw new Error('Ollama offline'); }
  });
  await store.addDocuments([doc('known')]);
  await assert.rejects(store.similaritySearch('query'), /Query embedding unavailable/);
});

test('an invalid batch is rejected atomically', async () => {
  const store = new SimpleVectorStore(provider([[1, 0], [NaN, 0]]));
  await assert.rejects(store.addDocuments([doc('valid'), doc('invalid')]), /finite numeric/);
  assert.equal(store.allVectors.length, 0);
});

test('invalid embedding counts, dimensions, zero norms and non-finite values are rejected', async () => {
  for (const vectors of [[], [[1, 0], [0, 1]], [[0, 0]], [[]], [[Infinity, 0]]]) {
    const store = new SimpleVectorStore(provider(vectors));
    await assert.rejects(store.addDocuments([doc('input')]));
    assert.equal(store.allVectors.length, 0);
  }
  const store = new SimpleVectorStore(provider([[1, 0]], [1, 0, 0]));
  await store.addDocuments([doc('input')]);
  await assert.rejects(store.similaritySearch('query'), /dimension changed/);
});

test('a later dimension-changing batch does not damage the existing index', async () => {
  let vectors = [[1, 0]];
  const store = new SimpleVectorStore({
    embedDocuments: async () => vectors,
    embedQuery: async () => [1, 0]
  });
  await store.addDocuments([doc('original')]);
  vectors = [[1, 0, 0]];
  await assert.rejects(store.addDocuments([doc('wrong model')]), /dimension changed/);
  assert.deepEqual((await store.similaritySearch('query')).map(d => d.pageContent), ['original']);
});

test('empty indexes and zero limits return no matches without contacting embeddings', async () => {
  const unavailable = {
    embedDocuments: async () => { throw new Error('unexpected call'); },
    embedQuery: async () => { throw new Error('unexpected call'); }
  };
  const empty = new SimpleVectorStore(unavailable);
  assert.deepEqual(await empty.similaritySearch('query'), []);
  await empty.addDocuments([]);
  const store = new SimpleVectorStore({ ...unavailable, embedDocuments: async () => [[1, 0]] });
  await store.addDocuments([doc('input')]);
  assert.deepEqual(await store.similaritySearch('query', 0), []);
  await assert.rejects(store.similaritySearch('query', -1), /non-negative integer/);
});

test('large finite coordinates produce a meaningful cosine ranking', async () => {
  const store = new SimpleVectorStore(provider([[1e200, 0], [0, 1e200]], [1e200, 0]));
  await store.addDocuments([doc('aligned'), doc('perpendicular')]);
  assert.equal((await store.similaritySearch('query', 1))[0].pageContent, 'aligned');
});


# AutoCritic

An early local code-review prototype using Next.js, LangChain and Ollama. It accepts code, generates structured critiques and stores feedback and earlier examples locally.

**Status:** historical prototype. Development continued in [AutoAgent](https://github.com/Dolvido/AutoAgent), which adds ticket and code-inspection experiments. This repository remains useful for the earlier critique-and-feedback design.

## What is implemented

- React code input and structured critique cards.
- Local Ollama calls through LangChain, exposed through Next.js API routes.
- SQLite storage for critiques and feedback.
- An exact in-memory cosine index using Ollama embeddings, with JSON persistence.
- Experimental feedback analysis that proposes few-shot examples.

The vector store is a small cosine-search implementation, not FAISS. Feedback-based adaptation is implemented as experimental routines; improved review correctness has not been established by an evaluation.

## Local setup

The original application targets Node.js 18+ with local Ollama. The focused tests below require **Node.js 22.18+** for built-in TypeScript stripping. SQLite dependencies may require platform build tools.

```bash
git clone https://github.com/Dolvido/AutoCritic.git
cd AutoCritic
npm ci
ollama pull codellama
ollama pull llama3
npm run dev
```

Start Ollama before launching the app. Open [localhost:3000](http://localhost:3000). The default service address is `http://localhost:11434`; the Windows `setup-ollama.ps1` script provides an interactive model-download helper.

The example store writes to `data/vectors/vector-data.json` and re-embeds stored text on startup. It rejects embedding failures and invalid vectors. The main critique path reports model/retrieval errors instead of returning fabricated sample critiques.

## Focused checks

```bash
npm run test:vector-store
```

The tests exercise cosine ordering, embedding-provider failures, invalid dimensions and atomic insertion using deterministic test providers. They require no models or application dependencies. They do not establish a successful full application build, live inference or measured review quality.

## Limitations

This is a local development prototype with legacy dependencies and experimental codebase-review paths. Filesystem access, model availability, storage behavior and review quality need further work before wider deployment. Source snippets and feedback are stored locally and may appear in application logs.

The original requirements document describes design goals, including technologies and features that were not all implemented. This README describes the current source.

## License

No standalone license file has been specified for this repository.


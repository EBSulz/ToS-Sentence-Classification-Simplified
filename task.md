
# Semantic Similarity Search Service
A scientific approach is necessary. Defining the methodology, doing experiments, explaining the results in the report and the progression of each of these experiments is a MUST have and tasks without them wont be considered.

## Objective

Build a small, production-style Python service that:

- Indexes a corpus of sentences  
- Supports semantic similarity search (e.g., "find the most similar sentences")  
- Optionally includes a classification component to detect "unfair clauses"  
- Exposes functionality via REST APIs  
- Is containerized using Docker  

**Bonus (optional):**
- Provide a minimal user interface to interact with the service  

---

## Deliverables

Please submit:

- A Python-based service (e.g., FastAPI or Flask)
- Source code with clear structure and instructions to run
- A `Dockerfile` or a `docker-compose.yaml` to build and run the service
- A short technical report (PDF or Word) explaining:
  - your approach
  - key design decisions
  - trade-offs and limitations

---

## Dataset Description
https://arxiv.org/pdf/1805.01217

---

## Rules / Constraints
	- No RAG / No generative LLM usage: do not call ChatGPT/Claude/etc, and do not do retrieval + generation.
	- Any other methodology is more than welcome.
	- Keep it small and focused - optimize for clarity and engineering quality.
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Example sentence
sentence = ["What are the documents uploaded in the database?"]
embeddings = model.encode(sentence)

print(f"Embedding dimension: {embeddings.shape}")

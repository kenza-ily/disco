# Embeddings Cache Directory

This directory stores cached embedding vectors for analysis notebooks.

## Structure

```
3_embeddings/
├── IAM_mini/
│   ├── Pa_embeddings_text-embedding-3-large_20260129_193045.json
│   ├── Pb_embeddings_text-embedding-3-large_20260129_193045.json
│   └── Pc_embeddings_text-embedding-3-large_20260129_193045.json
├── DocVQA_mini/
│   └── ...
└── ...
```

## File Format

Each JSON file contains:
```json
{
  "ground_truths": {
    "text_string": [embedding_vector]
  },
  "predictions": {
    "sample_id": {
      "model_name": [embedding_vector]
    }
  },
  "_metadata": {
    "dataset": "IAM_mini",
    "phase": "Pa",
    "model": "text-embedding-3-large",
    "timestamp": "20260129_193045"
  }
}
```

## Usage

Embeddings are automatically:
1. **Loaded** when a notebook runs (if cached files exist)
2. **Generated** on-the-fly if not cached (with progress bar)
3. **Saved** at the end of notebook execution for future use

## Note

These files are ignored by git (large JSON files).
To regenerate embeddings, delete the corresponding JSON files.

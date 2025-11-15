## To run in dev
1. pip install -r embeddings/requirements.txt
2. cd embeddings  && uvicorn embed_specs_to_chroma:app --host 0.0.0.0 --port 8000 --reload
3. curl -X POST "http://localhost:8000/embed" \
  -F "collection_name=monitor_guide" \
  -F "file=@/path/to/your_file.pdf"


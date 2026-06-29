source .venv/bin/activate
uvicorn llm_server:app --host 0.0.0.0 --port 8002

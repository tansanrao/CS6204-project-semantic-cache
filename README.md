Tool setup

Setup environment - install dependencies
```bash
uv sync --frozen --python 3.11
```

Setup LLM and DB stack
```bash
docker compose -f db-stack.yaml up -d
docker compose -f llm-stack.yaml up -d
```

Run script
```bash
uv run main.py
```
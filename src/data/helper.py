import os
from pathlib import Path

from openai import OpenAI

def _parse_env_file(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def _load_api_key() -> str:
    env_path = Path(__file__).parent / ".env"
    parsed = _parse_env_file(env_path)
    api_key = os.getenv("PROXY_API_KEY") or parsed.get("PROXY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PROXY_API_KEY not set. Provide it in the environment or src/.env"
        )
    return api_key



MODEL_NAME = "openai/gpt-oss-20b"
PROXY_BASE_URL = "https://openrouter.ai/api/v1"
CHECKPOINT_EVERY = 10  # rows
TEMPERATURE = 0.4

PROXY_API_KEY = _load_api_key()
client = OpenAI(base_url=PROXY_BASE_URL, api_key=PROXY_API_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for the web to get current information about anything",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
def llm_with_tool_call(prompt: str, tool_call_response: str) -> str | None:
    """
    Issue a tool-call-driven chat completion and return the final assistant content.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant, you always respond to user queries"
                "You must use the web_search tool call to answer user queries."
                "Always do EXACTLY ONE web_search call. If needed, only summarize the tool call responses for the user query"
                "Write a concise response using information from the tool call response, keep the response short — no more than ~10 sentences (preferably fewer)"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        max_tokens=1024,
        stream=False,
        temperature=TEMPERATURE
    )
    response_message = response.choices[0].message

    if response_message.role == "assistant":
        try:
            tool_call_id = response_message.tool_calls[0].id
        except Exception:
            tool_call_id = None

        messages.extend(
            [
                response_message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": "web_search",
                    "content": tool_call_response,
                },
            ]
        )

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            stream=False,
            temperature=TEMPERATURE
        )
        return final_response.choices[0].message.content

    return None


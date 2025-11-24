"""
Reads `data/data.csv`, calls an LLM for each `tool_call_response` entry (parallelized),
and writes `ttl_bucket,prompt,responses` rows to `data/ai_response_data.csv`.
`responses` is a list of `(tool_call_response, ai_response)` tuples.
If `data/ai_response_data.csv` already exists, previously completed rows are reused
and only missing tool responses are computed. Periodic checkpoints persist
intermediate progress but never write partially filled rows.
"""

# Modified requirements:
# 1. csv structure: ttl_bucket,prompt,responses, where responses is a list of
#    tuples (tool_call_response, ai_response)
# 2. add checkpointing: every N completed rows, write intermediate results
# 3. when resuming, detect partially processed prompts and complete remaining

import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import datetime

import pandas as pd
from openai import OpenAI


MODEL_NAME = "openai/gpt-oss-20b"
PROXY_BASE_URL = "https://openrouter.ai/api/v1"
CHECKPOINT_EVERY = 10  # rows
TEMPERATURE = 0.4


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


def responses_complete(responses: list) -> bool:
    return bool(responses) and all(r is not None for r in responses)


def load_existing_outputs(output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if "responses" in existing_df.columns:
            existing_df["responses"] = existing_df["responses"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        else:
            existing_df["responses"] = [[] for _ in range(len(existing_df))]
        # Keep only fully completed rows; drop partials to avoid persisting Nones.
        existing_df = existing_df[existing_df["responses"].apply(responses_complete)]
        return existing_df.reset_index(drop=True)
    return pd.DataFrame(columns=["ttl_bucket", "prompt", "responses"])


def load_input_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["tool_call_response"] = df["tool_call_response"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


def build_work_items(
    df: pd.DataFrame, completed_lookup: dict[tuple, List[Tuple[str, str]]]
) -> tuple[list[tuple], dict[tuple, List[Tuple[str, str] | None]]]:
    """
    Build work items and initialize a responses map.
    Work item format: (row_key, response_idx, prompt, tool_call_response)
    Row keys are (ttl_bucket, prompt) pairs to tolerate reordering.
    """
    work_items: list[tuple[tuple, int, str, str]] = []
    responses_map: dict[tuple, List[Tuple[str, str] | None]] = {}

    for _, row in df.iterrows():
        row_key = (row["ttl_bucket"], row["prompt"])
        tool_responses = row["tool_call_response"]

        if row_key in completed_lookup:
            # Already fully processed; skip creating work.
            continue

        responses_map[row_key] = [None] * len(tool_responses)
        for resp_idx, tool_resp in enumerate(tool_responses):
            work_items.append((row_key, resp_idx, row["prompt"], tool_resp))

    return work_items, responses_map


def write_checkpoint(
    output_path: Path,
    completed_rows: list[dict],
) -> None:
    pd.DataFrame(completed_rows).to_csv(output_path, index=False)


def main() -> None:

    # input_path = Path("data/sample.csv")
    # output_path = Path("data/ai_response_sample.csv")

    input_path = Path("../data/data.csv")
    output_path = Path("../data/ai_response_data.csv")
    
    input_df = load_input_data(input_path)
    existing_df = load_existing_outputs(output_path)

    completed_rows = [
        {"ttl_bucket": row["ttl_bucket"], "prompt": row["prompt"], "responses": row["responses"]}
        for _, row in existing_df.iterrows()
    ]
    completed_lookup = {(row["ttl_bucket"], row["prompt"]): row["responses"] for row in completed_rows}

    work_items, responses_map = build_work_items(input_df, completed_lookup)
    max_worker_count = 30
    print(f'Total work items to process: ({max_worker_count} workers)', len(work_items))

    # Parallelize LLM calls across all pending tool_call_responses.
    with ThreadPoolExecutor(max_workers=max_worker_count) as executor:
        future_to_key = {
            executor.submit(llm_with_tool_call, prompt, tool_resp): (
                row_key,
                resp_idx,
                prompt,
                tool_resp,
            )
            for row_key, resp_idx, prompt, tool_resp in work_items
        }
        completed_rows_since_checkpoint = 0
        for future in as_completed(future_to_key):
            row_key, resp_idx, _prompt, tool_resp = future_to_key[future]
            try:
                ai_resp = future.result()
                responses_map[row_key][resp_idx] = (tool_resp, ai_resp)
            except Exception as exc:  # Keep failures visible in output data.
                responses_map[row_key][resp_idx] = (tool_resp, f"ERROR: {exc}")

            if responses_complete(responses_map[row_key]):
                completed_rows.append(
                    {
                        "ttl_bucket": row_key[0],
                        "prompt": row_key[1],
                        "responses": responses_map[row_key],
                    }
                )
                # print(f'Completed row: {row_key[1]}')
                completed_rows_since_checkpoint += 1
                responses_map.pop(row_key, None)

                if completed_rows_since_checkpoint >= CHECKPOINT_EVERY:
                    print(f'Checkpointing {len(completed_rows)} completed rows...')
                    write_checkpoint(output_path, completed_rows)
                    completed_rows_since_checkpoint = 0

    # Final flush of all completed rows.
    write_checkpoint(output_path, completed_rows)


if __name__ == "__main__":
    main()

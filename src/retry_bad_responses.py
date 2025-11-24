"""
Reload `data/ai_response_data.csv`, find bad AI responses, retry in parallel with
checkpointing, and rewrite the file.

Bad responses are those that are empty strings or contain the substring "<|start>".
For each bad response we call `llm_with_tool_call(prompt, tool_call_response)`,
replace the entry in-memory, and persist the full CSV back to disk periodically.
"""

from __future__ import annotations

import ast
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from collect_ai_response_train_data import llm_with_tool_call


DATA_PATH = Path("../data/ai_response_data.csv")
BAD_PATTERN = re.compile(r"<\|start")
CHECKPOINT_EVERY = 10  # completed retries
MAX_WORKERS = 30


def is_bad_response(resp: str | None) -> bool:
    if resp is None:
        return True
    # if not isinstance(resp, str):
    #     return True
    if resp.strip() == '':
        return True
    # if BAD_PATTERN.search(resp):
    #     return True
    return False


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["responses"] = df["responses"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


def write_checkpoint(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_work(df: pd.DataFrame) -> tuple[list[tuple], dict[int, List[Tuple[str, str | None]]]]:
    work_items: list[tuple] = []
    responses_map: dict[int, List[Tuple[str, str | None]]] = {}
    for idx, row in df.iterrows():
        responses: List[Tuple[str, str | None]] = row["responses"]
        responses_map[idx] = list(responses)
        for resp_idx, (tool_resp, ai_resp) in enumerate(responses):
            if is_bad_response(ai_resp):
                work_items.append((idx, resp_idx, row["prompt"], tool_resp))
    return work_items, responses_map


def retry_bad_entries(df: pd.DataFrame) -> pd.DataFrame:
    work_items, responses_map = build_work(df)
    if not work_items:
        print("No bad responses found.")
        return df

    print(f"Retrying {len(work_items)} bad responses using {MAX_WORKERS} workers...")
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(llm_with_tool_call, prompt, tool_resp): (idx, resp_idx, tool_resp)
            for idx, resp_idx, prompt, tool_resp in work_items
        }
        for future in as_completed(future_to_item):
            idx, resp_idx, tool_resp = future_to_item[future]
            try:
                ai_resp = future.result()
            except Exception as exc:
                ai_resp = f"ERROR: {exc}"
            responses_map[idx][resp_idx] = (tool_resp, ai_resp)

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                # Apply updates to df and checkpoint.
                for row_idx, responses in responses_map.items():
                    df.at[row_idx, "responses"] = responses
                write_checkpoint(df, DATA_PATH)
                print(f"Checkpointed after {completed} retries...")

    # Final application of updates and save.
    for row_idx, responses in responses_map.items():
        df.at[row_idx, "responses"] = responses
    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = load_dataframe(DATA_PATH)
    df = retry_bad_entries(df)
    write_checkpoint(df, DATA_PATH)
    print("Retry pass complete.")


if __name__ == "__main__":
    main()

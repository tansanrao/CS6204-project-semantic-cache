import ast
import pandas as pd

from openai import OpenAI

def get_train_test(path):
    df = pd.read_csv(path)

    # Convert tool_call_response string → Python list
    df["tool_call_response"] = df["tool_call_response"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # -------- Stratified 80/20 Split --------
    train_df = (
        df.groupby("ttl_bucket", group_keys=False)
        .apply(lambda x: x.sample(frac=0.8, random_state=42))
    )

    test_df = df.drop(train_df.index)

    # Shuffle training set
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

MODEL_NAME = "openai/gpt-oss-20b"
# PROXY_BASE_URL = "http://localhost:8000/v1"
PROXY_BASE_URL = "https://openrouter.ai/api/v1"
PROXY_API_KEY = "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"
client = OpenAI(
    base_url=PROXY_BASE_URL, 
    api_key=PROXY_API_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for the web to get current information about anything",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]}}}]
def llm_with_tool_call(prompt, tool_call_response):    
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant, you always respond to user queries"
            "You must use the web_search tool call to answer user queries."
            "Do ONLY ONE web_search call. If needed, only summarize the tool call responses for the user query"
            "Write a concise response using information from the tool call response, keep the response short — no more than ~10 sentences (preferably fewer)"
        )},
        {"role": "user", "content": f"{prompt}"}]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        max_tokens=1024,
        stream=False,
    )
    response_message = response.choices[0].message
    print(response)

    if(response_message.role == "assistant"):
        try: tool_call_id = response_message.tool_calls[0].id
        except: tool_call_id = None

        tool_call_extension = [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "web_search",
                "content": tool_call_response
            }
        ]
        print(tool_call_extension)
        messages.extend(tool_call_extension)

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            stream=False
        )

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            stream=False
        )  
        llm_answer = final_response.choices[0].message.content
        print("Final response:", final_response)
        return llm_answer
    
    return None


from helper import llm_with_tool_call

# prompt = "List a few mathematical truths that are eternally valid"
# tool_call_response = "1. 1 + 1 = 2. 2. The sum of the angles in a triangle is 180° (in Euclidean geometry). 3. A prime number is divisible only by 1 and itself. 4. The square root of 4 is always 2. 5. π (pi) is an irrational number."

# prompt = "Balanced BST (e.g., AVL, Red-Black) search time"
# tool_call_response = "O(log n)"

prompt = "Create a structured weekly status for insider buys and sells"
tool_call_response = "Insiders: buys/sells ratio 0.9:1; activity concentrated in Tech"

# prompt = "Condense today’s activity in API error rates, latency SLOs, and incidents today (aggregate) into a brief"
# tool_call_response = "API errors +1.79%; p95 latency 584.02 ms; incidents 4"


print(llm_with_tool_call(prompt, tool_call_response))

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "web_search",
#             "description": "Search for the web to get current information about anything",
#             "parameters": {
#                 "type": "object",
#                 "properties": {"query": {"type": "string"}},
#                 "required": ["query"]}}}]

# PROXY_BASE_URL = "https://openrouter.ai/api/v1"
# PROXY_API_KEY = "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"
# MODEL_NAME = "openai/gpt-oss-20b"
# prompt = "List a few mathematical truths that are eternally valid"

# client = OpenAI(
#         base_url=PROXY_BASE_URL, 
#         api_key=PROXY_API_KEY)
    
# messages = [
#     {"role": "system", "content": (
#         "You are a helpful assistant. "
#         "You must use the web_search tool call to answer user queries."
#         "Write a concise response using information from the tool call response, keep the response short — no more than ~10 sentences (preferably fewer)"
#     )},
#     {"role": "user", "content": f"{prompt}"}]

# response = client.chat.completions.create(
#     model=MODEL_NAME,
#     messages=messages,
#     tools=tools,
#     max_tokens=1024,
#     stream=False,
# )
# print(response)
# response_message = response.choices[0].message
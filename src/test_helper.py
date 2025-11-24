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

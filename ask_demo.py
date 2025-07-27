from openai import OpenAI

def ask_openai(base_url="http://localhost:5000/v1", model_name="Qwen3-8B", stop_str = ["</answer>", "</code>"]):

    client = OpenAI(
    base_url=base_url,
    api_key="EMPTY",
    )

    question = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
    retool_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in ```output{result}```) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.\n\n**Code Format:**\nEach code snippet is wrapped with\n```python\ncode snippet\n```\n**Answer Format:**\nThe last part of your response should be:\n<answer>\\boxed{'The final answer goes here.'}</answer>\n\n**User Question:**\n{question}\n\n**Assistant:**\n"
    completion = client.completions.create(
        model=model_name,
        max_tokens=512,
        temperature=1.0,
        stop=stop_str,
        prompt = retool_prompt.replace("{question}", question)

        )

    print(completion)
    
    
def ask_openai_stream(base_url="http://localhost:5000/v1", model_name="Qwen3-8B", stop_str = ["</answer>", "</code>"]):

    client = OpenAI(
    base_url=base_url,
    api_key="EMPTY",
    )

    question = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
    retool_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in ```output{result}```) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.\n\n**Code Format:**\nEach code snippet is wrapped with\n```python\ncode snippet\n```\n**Answer Format:**\nThe last part of your response should be:\n<answer>\\boxed{'The final answer goes here.'}</answer>\n\n**User Question:**\n{question}\n\n**Assistant:**\n"
    completion = client.completions.create(
        model=model_name,
        max_tokens=2048,
        temperature=1.0,
        stop=stop_str,
        stream = True,
        prompt = retool_prompt.replace("{question}", question)

        )
    output = ""
    for response in completion:
        print(response)
        output += response.model_dump()["choices"][0]["text"]
    print("="*9)
    print(output)
    
def chat_openai(base_url="http://localhost:5000/v1", model_name="Qwen3-8B", stop_str = ["</answer>", "</code>"]):

    client = OpenAI(
    base_url=base_url,
    api_key="EMPTY",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for, e.g. 'San Francisco'"
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'"
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city", "state", "unit"]
            }
        }
    }]

    question = "count the number of 0 in binary of ascii code \"9pwgoibnzUsyf89tg34qklnwf\""
    retool_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in ```output{result}```) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.\n\n**Code Format:**\nEach code snippet is wrapped with\n```python\ncode snippet\n```\n**Answer Format:**\nThe last part of your response should be:\n<answer>\\boxed{'The final answer goes here.'}</answer>\n\n**User Question:**\n{question}\n\n**Assistant:**\n"

    
    messages = [{
        "role": "user",
        "content": retool_prompt.replace("{question}", question)
    }]
    chat_completion = client.chat.completions.create(
        messages=messages,
        max_tokens=4096,
        model=model_name,
        tools=tools,
        stop=stop_str)

    print(chat_completion)
    
def chat_openai_stream(base_url="http://localhost:5000/v1", model_name="Qwen3-8B", stop_str = ["</answer>", "</code>"]):

    client = OpenAI(
    base_url=base_url,
    api_key="EMPTY",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for, e.g. 'San Francisco'"
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'"
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city", "state", "unit"]
            }
        }
    }]

    question = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
    retool_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in ```output{result}```) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.\n\n**Code Format:**\nEach code snippet is wrapped with\n```python\ncode snippet\n```\n**Answer Format:**\nThe last part of your response should be:\n<answer>\\boxed{'The final answer goes here.'}</answer>\n\n**User Question:**\n{question}\n\n**Assistant:**\n"

    
    messages = [{
        "role": "user",
        "content": retool_prompt.replace("{question}", question)
    }]

    
    chat_completion = client.chat.completions.create(
        messages=messages,
        max_tokens=4096,
        temperature=1.0,
        model=model_name,
        tools=tools,
        stream=True,
        stop=stop_str)

    output = ""
    for response in chat_completion:
        print(response)
        output += response.model_dump()["choices"][0]["delta"]["content"]
    print("="*9)
    print(output)


if __name__ == '__main__':
    base_url="http://localhost:5000/v1"
    model_name="/workspace/models/Qwen/Qwen3-8B"
    stop_str = ["</answer>"]
    # ask_openai_stream(base_url, model_name, stop_str)
    # ask_openai(base_url, model_name, stop_str)
    # chat_openai_stream(base_url, model_name, stop_str)
    chat_openai(base_url, model_name, stop_str)

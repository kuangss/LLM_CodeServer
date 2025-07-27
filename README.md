# vLLM Code Server

A proxy server that make your LLM support auto code execution.

## Features
- **Code executor:** Auto code execution via PythonExecutor for LLM
- **OpenAI compatible:** Compatible with OpenAI API, which means it can serve popular apps like chatbox
- **Streaming mode:** Implements streaming responses
- **Chat/text completions:** Handles both chat completions and text completions

## Requirements
```bash
pip install flask requests openai python-executor
```

## Usage

### Start the server
```bash
python vllm_code_serve.py --target_url http://localhost:3000 \
  --max_func_call 5 \
  --func_call_timeout 30 \
  --func_call_mode jupyter
```

### Available Arguments
```text
--max_func_call        Maximum number of function calls (default: 5)
--func_call_mode       Mode for function calls: "jupyter" or "standard" (default: jupyter)
--func_call_timeout    Timeout for function calls in seconds (default: 30)
--target_url           URL of the target model server (default: http://localhost:3000)
```

### Request examples
- streaming responses get the code and execute it while text generating, see `ask_openai_stream` (text completions) and `chat_openai_stream` (chat completions) in `ask_demo.py` as an example.
- non-streaming response extract the last code block, see `ask_openai` (text completions) and `chat_openai` (chat completions) in `ask_demo.py`.

## Endpoints
- `/v1/chat/completions`
- `/v1/completions`

## Notes
- Requires the target model server to be running
- DEBUG and LOG flags can be enabled for detailed output
- Supports both standard and Jupyter-like code execution modes
- Each code snippet is wrapped with```python\ncode snippet\n```
- The last part of your response should be:<answer>\\boxed{'The final answer goes here.'}</answer>

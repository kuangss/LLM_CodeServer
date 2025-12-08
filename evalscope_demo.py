from evalscope import TaskConfig, run_task
task_cfg = TaskConfig(
    model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    # model='Qwen3-8B',
    api_url='http://127.0.0.1:5002/v1/chat/completions',
    api_key='EMPTY',
    eval_type='openai_api',
    datasets=[
    # 'math_500',  # 数据集名称
    'gsm8k',
    ],
    dataset_args={
        "gsm8k":{
        'prompt_template': "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in ```output{result}```) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports and print function.\n\n**Code Format:**\nEach code snippet is wrapped with\n```python\ncode snippet\n```\n**Answer Format:**\nThe last part of your response should be:\n<answer>\\boxed{'The final answer goes here.'}</answer>\n\n**User Question:**\n{query}\n\n**Assistant:**\n"}
    },
    eval_batch_size=2,
    generation_config={
        'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'n': 2,  # 每个请求产生的回复数量
        'chat_template_kwargs': {'enable_thinking': False},  # 关闭思考模式
        'stream':True
    },
    timeout=60000,  # 超时时间
    stream=True,  # 是否使用流式输出
    limit=10,  # 设置为1000条数据进行测试
    # use_cache="/workspace/vllm/outputs/20250916_150439"
)

run_task(task_cfg=task_cfg)

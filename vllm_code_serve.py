from flask import Flask, request, Response
import requests
import datetime
import json
import argparse
import time
from openai import OpenAI
from utils import get_end_of_string, text_to_stream, concate_chunks, response_to_chunk, get_nostream_content
from python_executor import PythonExecutor
from parser import extract_jupyter_like_program, extract_program, process_string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multi_queue import ParallelGenerators

num_threads = 16
CHUNK_SIZE = 100
DEBUG = False
LOG = False
GMT_FORMAT='%a, %d %b %Y %H:%M:%S GMT'

END_OF_CHUNKS = "data: [DONE]\n\n"

app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_func_call", default=5, type=int)
    parser.add_argument("--max_token_len", default=32768, type=int)
    parser.add_argument("--func_call_mode", default="jupyter", type=str)
    parser.add_argument("--func_call_timeout", default=30, type=int)
    parser.add_argument("--port", default=5002, type=int)
    parser.add_argument("--target_url", default="https://api.siliconflow.cn", type=str)
    parser.add_argument("--stop_strs", default=["</answer>", "</code>"], type=str)
    
    args = parser.parse_args()
    print(f"args detail:")
    print(args)
    return args


def get_exec_result(response_text, is_lack_token):

    assert "```python" in response_text
    if args.func_call_mode == "jupyter":
        program = extract_jupyter_like_program(response_text)
    else:
        program = extract_program(response_text)
    executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=args.func_call_timeout)

    # Execute the remain prompts
    remain_results = executor.batch_apply([program])
    res, report = remain_results[0]

    exec_result = res if res else report
    # run out of python exe
    if is_lack_token:
        exec_result += f"\n\n[SYSTEM]\nYou have exceeded the allowed number of code executions. You can no longer write or run code. Please continue solving the problem using your reasoning and analytical skills."
    exec_result = f"\n```output\n{exec_result}\n```\n"

    return exec_result


def chat_complete_one(messages, req_method, req_headers, req_dict, req_args):
    message = ""
    for epoch in range(args.max_func_call):

        query = "".join([message["role"] + ":\n" + message["content"] + '\n' for message in messages])
        req_dict['prompt'] = query
        req_dict["messages"] = messages
        # req_dict.pop("messages", None)
        req_json = json.dumps(req_dict, ensure_ascii=False)
        req_bytes = bytes(req_json, encoding='utf-8')
        req_headers["Content-Length"] = str(len(req_bytes))

        target_route = args.target_url + "/v1/chat/completions" if epoch == 0 else args.target_url + "/v1/completions"
        # send modified request
        response = requests.request(
            method=req_method,
            url=target_route,
            headers=req_headers,
            json=req_dict,
            params=req_args,
        )
        
        print(response)
        if epoch == 0:
            prompt_tokens = response.json()["usage"]["prompt_tokens"]
            template = response

        _, response_text_dict = get_nostream_content(response.json()["choices"][0])
        flag, response_text = process_string(response_text_dict)
        # flag = 1: with complete code, flag = 0: no code, flag = -1: with incomplete code
        
        # add content to the messages and response
        messages.append({"role": "assistant", "content": response_text})
        message += response_text
        # Execute the remain prompts
        if flag == 1:
            # add user content
            is_lack_token = epoch >= args.max_func_call - 2
            exec_result = get_exec_result(response_text, is_lack_token)
            messages.append({"role":"user", "content":exec_result})
            message += exec_result
        else:
            # break if no code is to exec
            if "finish_reason" in response.json()["choices"][0].keys() and response.json()['choices'][0]['finish_reason'] == "stop" and "stop_reason" in response.json()["choices"][0].keys() and response.json()['choices'][0]['stop_reason'] in ["</answer>"]:
                messages[-1]["content"] += response.json()['choices'][0]['stop_reason']
                message += response.json()['choices'][0]['stop_reason']
            break
            
    # complete the response
    response_content_dict = template.json()
    response_content_dict["choices"] = [{"index": 0, "message": {"role": "assistant", "content": message}, 'logprobs': None, 'finish_reason': None, 'stop_reason': None}]
    response_content_dict["usage"]["completion_tokens"] = template.json()["usage"]["total_tokens"] - prompt_tokens
    response_content_dict["usage"]["prompt_tokens"] = prompt_tokens

    return template.status_code, template.headers, response_content_dict


@app.route('/v1/chat/completions', methods=['GET', 'POST', 'PUT', 'DELETE'])
def chat_completions(path=None):
    rsp_headers = {'date': datetime.datetime.now(datetime.timezone.utc).strftime(GMT_FORMAT), 
                   'server': 'uvicorn', 
                   'content-type': 'text/event-stream; charset=utf-8', 
                   'connection': 'close'}
    
    # 获取请求数据
    req_method = request.method
    req_headers = {key:value for key, value in request.headers.items()}
    req_headers["Host"] = args.target_url.split("//")[-1]
    req_headers["Accept-Encoding"] = "identity"
    req_dict = request.get_json()
    # req_dict["stop"] = args.stop_strs
    # req_dict["priority"] = 1
    req_args = request.args

    n_sample = req_dict["n"] if "n" in req_dict.keys() else 1
    req_dict["n"] = 1
    req_dict['max_tokens'] = req_dict["max_tokens"] if "max_tokens" in req_dict else 8192
    messages = [request.get_json()["messages"] for _ in range(n_sample)]

    if "stream" in req_dict.keys() and req_dict["stream"] == True:
        def generate():
            
            def process_single_sample(index):
                """处理单个样本的生成器函数"""
                template = ""
                is_first_chunk = True
                
                is_finished = 0
                response_text = ""
                received_texts = [["" for i in range(n_sample)] for j in range(args.max_func_call)]

                for epoch in tqdm(range(args.max_func_call)):
                    if is_finished == 1:
                        print(f"finished{epoch}")
                        break

                    query = "".join([message["role"] + ":\n" + message["content"] + '\n' for message in messages[index]])
                    req_dict['prompt'] = query
                    req_dict["messages"] = messages[index]
                    # req_dict.pop("messages", None)
                    req_json = json.dumps(req_dict, ensure_ascii=False)
                    req_bytes = bytes(req_json, encoding='utf-8')
                    req_headers["Content-Length"] = str(len(req_bytes))

                    target_route = args.target_url + "/v1/chat/completions" if epoch == 0 else args.target_url + "/v1/completions"
                    
                    # send modified request
                    response = requests.request(
                        req_method,
                        target_route,
                        headers=req_headers,
                        json=req_dict,
                        params=req_args,
                        stream=True
                    )
                    response_encoding = response.encoding
                    received = ""
                    
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):
                        # data: {"id":"cmpl-7975264f5bf64aa3802a09f88e2a5772","object":"text_completion","created":1765100909,"model":"Qwen3-8B","choices":[{"index":1,"text":" so","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}
                        # data: {"id":"chatcmpl-9d80761cccbe4941891b5b0484b964b4","object":"chat.completion.chunk","created":1765101666,"model":"Qwen3-8B","choices":[{"index":1,"delta":{"content":","},"logprobs":null,"finish_reason":null}]}
                        if chunk:
                            received += chunk
                            is_finished, chunk_text, received, last_chunk, index_re = concate_chunks(received, response_encoding)
                            # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length, is_finished = -2: error
                            if last_chunk != "":
                                received_texts[epoch][index] += chunk_text
                                if is_first_chunk == True:
                                    is_first_chunk = False
                                    template = last_chunk
                            else:
                                continue
                            
                            if LOG:
                                print(chunk_text, end="")

                            # check if has code, concate the exec result by responding a chunk
                            has_code, response_text = process_string(received_texts[epoch][index])
                            # has_code = 1: with complete code, has_code = 0: no code, has_code = -1: with incomplete code

                            if has_code == 1:
                                # Execute the remain prompts
                                is_lack_token = epoch >= args.max_func_call - 2
                                exec_result = get_exec_result(response_text, is_lack_token)

                                # add response and exec_result to messages
                                messages[index].append({"role":"assistant", "content": response_text})
                                messages[index].append({"role":"user", "content": exec_result})
                                response_text += exec_result

                                if DEBUG:
                                    print(received_texts[epoch][index])
                                more_chunks = text_to_stream(chunk_text + exec_result, last_chunk, template, index)
                                
                                if is_finished == 1:
                                    more_chunks = more_chunks + text_to_stream("</answer>", last_chunk, template, index) + [
                                        END_OF_CHUNKS]
                                
                                for more_chunk in more_chunks:
                                    yield more_chunk
                                break

                            if is_finished == -2:
                                more_chunks = response_to_chunk(json.loads(last_chunk), CHUNK_SIZE)
                                return
                            else:
                                more_chunks = text_to_stream(chunk_text, last_chunk, template, index)
                            # end the loop, or continue by another request
                            if is_finished == 1:
                                more_chunks = more_chunks + text_to_stream("</answer>", last_chunk, template, index) + [END_OF_CHUNKS]
                            
                            for more_chunk in more_chunks:
                                yield more_chunk

                            if is_finished == 1:
                                break
                            if is_finished == -1 and has_code == 0:
                                return
            
            # 使用多线程处理多个样本
            if n_sample > 1:
                # 多样本时使用并行生成器
                with ParallelGenerators(max_workers=min(num_threads, n_sample)) as parallel:
                    for index in range(n_sample):
                        parallel.add_generator(process_single_sample, index)
                    
                    # 统一输出所有样本的结果
                    for chunk in parallel.start():
                        yield chunk
            else:
                # 单样本时直接处理
                for chunk in process_single_sample(0):
                    yield chunk

        return Response(
            generate(),
            status=200,
            headers=rsp_headers
        )

    
    else:
        pbar = tqdm(total=n_sample)
        pbar.set_description(f"Asking nostream")
        
        # 使用多线程替代多进程
        
        responses = [None] * n_sample
        response_status_code = ""
        response_headers  = ""
        completed_count = [0]
        lock = threading.Lock()
        
        def chat_complete_wrapper(index, messages, req_method, req_headers, req_dict, req_args):
            """包装 chat_complete_one 函数以在线程中执行"""
            nonlocal response_status_code, response_headers
            response_status_code, response_headers, result = chat_complete_one(messages, req_method, req_headers, req_dict, req_args)
            responses[index] = result
            with lock:
                completed_count[0] += 1
                pbar.update()
            return response_status_code, response_headers, result
        
        # 使用线程池执行
        with ThreadPoolExecutor(max_workers=min(num_threads, n_sample)) as executor:
            # 提交所有任务
            futures = []
            for index in range(n_sample):
                future = executor.submit(chat_complete_wrapper, index, messages[index], req_method, req_headers, req_dict, req_args)
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果或触发异常
                except Exception as e:
                    print(f"Error in thread: {e}")
                    raise
        
        # 确保所有响应都已收集
        if any(response is None for response in responses):
            raise RuntimeError("Not all responses were collected")

        response_content_dict = responses[0].copy()
        for index, response in enumerate(responses):
            if index == 0:
                continue
            this_choice = response["choices"][0]
            this_choice["index"] = index
            response_content_dict["choices"].append(this_choice)

        return Response(
            json.dumps(response_content_dict),
            status=response_status_code,
            headers=dict(response_headers)
        )


@app.route('/', methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/v1/completions', methods=['GET', 'POST', 'PUT', 'DELETE'])
def completions(path=None):
    # get requests
    req_method = request.method
    req_headers = {key:value for key, value in request.headers.items()}
    req_dict = request.get_json()
    rsp_headers = {'date': datetime.datetime.now(datetime.timezone.utc).strftime(GMT_FORMAT), 
                   'server': 'uvicorn', 
                   'content-type': 'text/event-stream; charset=utf-8', 
                   'connection': 'close'}

    if "stream" in req_dict.keys() and req_dict["stream"] == True :
        params = request.args
        def generate():
            response_text = ""
            query = req_dict['prompt']
            len_raw_request = int(req_headers["Content-Length"]) - len(query)
            template = ""
            is_finished = 0
            for epoch in range(args.max_func_call):
                if is_finished == 1:
                    return
                query += response_text
                req_dict['prompt'] = query
                req_headers["Content-Length"] = str(len_raw_request + len(req_dict['prompt']))
                response = requests.request(
                    req_method,
                    f"{args.target_url}/v1/completions",
                    headers=req_headers,
                    json=req_dict,
                    params=params,
                    stream=True
                )
                received = ""
                received_text = ""
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):

                    if chunk:
                        received += chunk
                        is_finished, received_text, chunk_text, received, last_chunk = concate_chunks(received_text, received)
                        # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length, is_finished = -2: error
                        if last_chunk != "" and epoch == 0:
                            template = last_chunk
                        
                        # check if has code, concate the exec result by responding a chunk
                        has_code, response_text = process_string(received_text)
                        # has_code = 1: with complete code, has_code = 0: no code, has_code = -1: with incomplete code


                        if has_code == 1:
                            # Execute the remain prompts
                            is_lack_token = epoch >= args.max_func_call - 2
                            exec_result = get_exec_result(response_text, is_lack_token)
                            response_text += exec_result
                            if DEBUG:
                                print(received_text)
                            remain_text = get_end_of_string(received_text)
                            more_chunks = text_to_stream(remain_text + exec_result, last_chunk, template)
                            # send last chunk first
                            more_chunks = [chunk] + more_chunks
                            if is_finished == 1:
                                more_chunks = more_chunks + text_to_stream("</answer>", last_chunk, template) + ["data: [DONE]\n\n"]
                            for more_chunk in more_chunks:
                                yield more_chunk
                            break

                        if LOG:
                            print(received_text)

                        # end the loop, or continue by another request
                        if is_finished == 1:
                            more_chunks = text_to_stream("</answer>", last_chunk, template)
                            more_chunks = [chunk] + more_chunks + ["data: [DONE]\n\n"]
                            for more_chunk in more_chunks:
                                yield more_chunk
                            break
                        elif is_finished == 0:
                            yield chunk
                        elif is_finished == -1:
                            pass
                

        return Response(
            generate(),
            status=200,
            headers=rsp_headers
        )

    
    else:
        query = req_dict['prompt']
        len_raw_query = len(query)
        for epoch in range(args.max_func_call):
            
            req_dict['prompt'] = query
            # edit content-length, otherwise the response_text is not used.
            req_headers["Content-Length"] = str(int(req_headers["Content-Length"]) + len(req_dict['prompt']) - len_raw_query)
            response = requests.request(
                req_method,
                f"{args.target_url}/v1/completions",
                headers=req_headers,
                json=req_dict,
                params=request.args,
            )
            flag, response_text = process_string(response.json()["choices"][0]['text'])
            # flag = 1: with complete code, flag = 0: no code, flag = -1: with incomplete code

            query += response_text

            if DEBUG:
                try:
                    print("="*9)
                    print(response.json()['choices'][0]['finish_reason'])#, response.json()['choices'][0]['stop_reason']
                    print("="*9)
                except Exception as e:
                    print(e)

            # Execute the remain prompts
            if flag == 1:
                is_lack_token = epoch >= args.max_func_call - 2
                exec_result = get_exec_result(response_text, is_lack_token)
                query += exec_result
            else:
                if 'finish_reason' in response.json()['choices'][0]:
                    if response.json()['choices'][0]['finish_reason'] == "stop":
                    # if response.json()['choices'][0]['finish_reason'] == "stop" and response.json()['choices'][0]['stop_reason'] in ["</answer>"]:
                    #     query += response.json()['choices'][0]['stop_reason']
                        break

            if LOG:
                print(query)

        response_content_dict = response.json()
        response_content_dict["choices"][0]["text"]=query
        return Response(
            json.dumps(response_content_dict),
            status=response.status_code,
            headers=dict(response.headers)
        )


if __name__ == '__main__':
    args = parse_args()

    app.run(debug=True, host="0.0.0.0", port=args.port)

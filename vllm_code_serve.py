from flask import Flask, request, Response
import requests
import datetime
import json
import argparse
import time
from openai import OpenAI
from multiprocessing import Process
from utils import get_end_of_string, text_to_stream, concate_chunks, get_chuncked_content
from python_executor import PythonExecutor
from parser import extract_jupyter_like_program, extract_program, process_string

CHUNK_SIZE = 100
DEBUG = False
LOG = True
GMT_FORMAT='%a, %d %b %Y %H:%M:%S GMT'

app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_func_call", default=5, type=int)
    parser.add_argument("--func_call_mode", default="jupyter", type=str)
    parser.add_argument("--func_call_timeout", default=30, type=int)
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--target_url", default="http://localhost:8888", type=str)
    
    args = parser.parse_args()
    print(f"args detail:")
    print(args)
    time.sleep(5)
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


@app.route('/v1/chat/completions', methods=['GET', 'POST', 'PUT', 'DELETE'])
def chat_completions(path=None):
    rsp_headers = {'date': datetime.datetime.now(datetime.timezone.utc).strftime(GMT_FORMAT), 
                   'server': 'uvicorn', 
                   'content-type': 'text/event-stream; charset=utf-8', 
                   'connection': 'close'}

    messages = request.get_json()["messages"]
    # 获取请求数据
    req_method = request.method
    req_headers = {key:value for key, value in request.headers.items()}
    req_dict = request.get_json()
    req_args = request.args

    if "stream" in req_dict.keys() and req_dict["stream"] == True:
        def generate():
            is_finished = 0
            # do the first time with chat mode
            # send modified request
            response = requests.request(
                req_method,
                f"{args.target_url}/v1/chat/completions",
                headers=req_headers,
                json=req_dict,
                params=req_args,
                stream=True
            )
            received = ""
            received_text = ""
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):

                if chunk:
                    received += chunk
                    is_finished, received_text, received, last_chunk = concate_chunks(received_text, received)
                    # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length
                    if last_chunk != "":
                        template = last_chunk

                    # check if has code, concate the exec result by responding a chunk
                    has_code, response_text = process_string(received_text)
                    # has_code = 1: with complete code, has_code = 0: no code, has_code = -1: with incomplete code


                    if has_code == 1:
                        # Execute the remain prompts
                        is_lack_token = False
                        exec_result = get_exec_result(response_text, is_lack_token)
                        response_text += exec_result

                        # add response and exec_result to messages
                        messages.append({"role":"assistant", "content": response_text})
                        if DEBUG:
                            print(received_text)
                        remain_text = get_end_of_string(received_text, template='```')
                        more_chunks = text_to_stream(remain_text + exec_result, template)
                        # send the last chunk first
                        more_chunks = [chunk] + more_chunks
                        if is_finished == 1:
                            more_chunks = more_chunks + text_to_stream("</answer>", template) + ["data: [DONE]\n\n"]
                        for more_chunk in more_chunks:
                            yield more_chunk
                        break

                    if LOG:
                        print(received_text)

                    # end the loop, or continue by another request
                    if is_finished == 1:
                        more_chunks = text_to_stream("</answer>", template)
                        more_chunks = [chunk] + more_chunks + ["data: [DONE]\n\n"]
                        for more_chunk in more_chunks:
                            yield more_chunk
                        break
                    elif is_finished == 0:
                        yield chunk
                    elif is_finished == -1:
                        if has_code == 0:
                            return
                        yield chunk

            query = "".join([message["role"] + ":\n" + message["content"] + '\n' for message in messages[:-1]])

            for epoch in range(1, args.max_func_call):
                if is_finished == 1:
                    return
                query +=  response_text
                req_dict['prompt'] = query
                req_dict['max_tokens'] = req_dict["max_tokens"] if "max_tokens" in req_dict else 8192
                req_json = json.dumps(req_dict, ensure_ascii=False)
                req_bytes = bytes(req_json, encoding='utf-8')
                req_headers["Content-Length"] = str(len(req_bytes)) # to ensure length of content
                response = requests.request(
                    "POST",
                    f"{args.target_url}/v1/completions",
                    headers=req_headers,
                    data=req_bytes,
                    stream=True
                )
                received_text = ""
                for chunk in response.iter_lines(chunk_size=CHUNK_SIZE, decode_unicode=True):

                    if chunk:
                        if chunk == "data: [DONE]":
                            is_finished = -1
                        else:
                            is_finished = 0
                            try:
                                chunk_text = get_chuncked_content(json.loads(chunk.split("data: ")[1])["choices"][0])
                            except:
                                yield chunk
                                is_finished = 1
                                break
                            received_text += chunk_text
                        # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length

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
                            more_chunks = text_to_stream(chunk_text + remain_text + exec_result, template)
                            
                            if is_finished == 1:
                                more_chunks = more_chunks + text_to_stream("</answer>", template) + [
                                    "data: [DONE]\n\n"]
                            for more_chunk in more_chunks:
                                yield more_chunk
                            break

                        if LOG:
                            # print(chunk)
                            print(received_text)

                        # end the loop, or continue by another request
                        if is_finished == 1:
                            more_chunks = text_to_stream(chunk_text + "</answer>", template)
                            more_chunks = more_chunks + ["data: [DONE]\n\n"]
                            for more_chunk in more_chunks:
                                yield more_chunk
                            break
                        elif is_finished == 0:
                            
                            more_chunks = text_to_stream(chunk_text, template)
                            for more_chunk in more_chunks:
                                yield more_chunk
                            
                        elif is_finished == -1:
                            if has_code == 0:
                                return
                            pass



        return Response(
            generate(),
            status=200,
            headers=rsp_headers
        )

    
    else:
        # combine LLM's and CI's response
        all_response = ""
        for epoch in range(args.max_func_call):
            # change message to add result of code
            req_dict["messages"] = messages
            # to ensure length of content
            req_json = json.dumps(req_dict, ensure_ascii=False)
            req_bytes = bytes(req_json, encoding='utf-8')
            req_headers["Content-Length"] = str(len(req_bytes)) # to ensure length of content
            # send modified request
            response = requests.request(
                req_method,
                f"{args.target_url}/v1/chat/completions",
                headers=req_headers,
                data=req_bytes,
                params=request.args,
            )
            flag, response_text = process_string(response.json()["choices"][0]["message"]["content"])
            # flag = 1: with complete code, flag = 0: no code, flag = -1: with incomplete code
            # cache the first response
            if epoch == 0:
                prompt_tokens = response.json()["usage"]["prompt_tokens"]
                template = response
            # add content to the messages and response
            messages.append({"role": "assistant", "content": response_text})
            all_response += response_text

            # Execute the remain prompts
            if flag == 1:
                # add user content
                is_lack_token = epoch >= args.max_func_call - 2
                exec_result = get_exec_result(response_text, is_lack_token)
                messages.append({"role":"user", "content":exec_result})
                all_response += exec_result
            else:
                # break if no code is to exec
                if response.json()['choices'][0]['finish_reason'] == "stop" and response.json()['choices'][0]['stop_reason'] in ["</answer>"]:
                    # print(response.json()['choices'][0]['stop_reason'] in ["</answer>"])
                    messages[-1]["content"] += response.json()['choices'][0]['stop_reason']
                    all_response += response.json()['choices'][0]['stop_reason']
                break
            if LOG:
                print(all_response)

        # complete the response
        response_content_dict = template.json()
        response_content_dict["choices"][0]["message"]["content"] = all_response
        response_content_dict["usage"]["completion_tokens"] = response.json()["usage"]["total_tokens"] - prompt_tokens
        response_content_dict["usage"]["prompt_tokens"] = prompt_tokens

        return Response(
            json.dumps(response_content_dict),
            status=response.status_code,
            headers=dict(response.headers)
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

    if "stream" in req_dict.keys() and req_dict["stream"] == True:
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
                        is_finished, received_text, received, last_chunk = concate_chunks(received_text, received)
                        # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length
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
                            more_chunks = text_to_stream(remain_text + exec_result, template)
                            # send last chunk first
                            more_chunks = [chunk] + more_chunks
                            if is_finished == 1:
                                more_chunks = more_chunks + text_to_stream("</answer>", template) + ["data: [DONE]\n\n"]
                            for more_chunk in more_chunks:
                                yield more_chunk
                            break

                        if LOG:
                            print(received_text)

                        # end the loop, or continue by another request
                        if is_finished == 1:
                            more_chunks = text_to_stream("</answer>", template)
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

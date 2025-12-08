
import random
import os
import numpy as np
import json
import re
DEBUG = False

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def messages2prompt(messages):
    query = ""
    for message in messages:
        query += '\n' + message["role"] + ": "
        query += '\n' + message["content"] + '\n'
    return query

def prompt2messages(prompt):
    
    content_indexes = []
    for key in PROMPT_KEYS:
        indexes = re.finditer(key, prompt)
        for index in indexes:
            content_indexes.append(index.span())

    sorted_content_indexes = sorted(content_indexes, key=lambda x: x[0])
    sorted_content_indexes.append([len(prompt), len(prompt)])

    contents = []
    for content_index_l, content_index_r in zip(sorted_content_indexes[:-1], sorted_content_indexes[1:]):
        title = prompt[content_index_l[0]:content_index_l[1]]
        content = prompt[content_index_l[1]:content_index_r[0]]
        contents.append({"role": role[title], "content": content})
        
    return {"messages": contents}


def get_nostream_content(data_dict, encoding='utf-8'):
    index = data_dict["index"] if "index" in data_dict.keys() else 0
    content = ""
    if "text" in data_dict.keys():
        content = data_dict["text"]
    elif "message" in data_dict.keys():
        if "reasoning_content" in data_dict["message"].keys() and data_dict["message"]["reasoning_content"] != None:
            content += "<think>\n" + data_dict["message"]["reasoning_content"] + "\n</think>\n"
        if "reasoning_content" in data_dict["message"].keys() and data_dict["message"]["content"] != None:
            content += data_dict["message"]["content"]
    else: 
        content = ""
    return index, content.encode(encoding).decode("utf-8")

def get_chuncked_content(data_dict, encoding="utf-8"):
    index = data_dict["index"]
    if "text" in data_dict.keys():
        content = data_dict["text"]
    elif "delta" in data_dict.keys():
        if "content" in data_dict["delta"] and data_dict["delta"]["content"] != None:
            content = data_dict["delta"]["content"]
        elif "reasoning_content" in data_dict["delta"] and data_dict["delta"]["reasoning_content"] != None:
            content = data_dict["delta"]["reasoning_content"]
        else:
            content = ""
    else: 
        content = ""
    return index, content.encode(encoding).decode("utf-8")


def concate_chunks(received_chunks, stop_strs, encoding="utf-8"):
    # is_finished = 1: end with </answer>, is_finished = 0: not finished, is_finished = -1: end with length, is_finished = -2: error
    last_chunk = ""
    flag = 0
    chunks = received_chunks.split('\n\n')
    chunk_text = ""
    index = -1
    data = ""
    
    if "data: [DONE]" in chunks[0]:
        flag = -1
        if DEBUG:
            print("="*9)
            print(chunks[0])
    else:
        if len(chunks) > 1:
            chunk = chunks[0]
            last_chunk = chunk
            try:
                choices = json.loads(chunk.split("data: ")[1])["choices"]
                if len(choices) > 0:
                    data = json.loads(chunk.split("data: ")[1])["choices"][0]
                    index, chunk_text = get_chuncked_content(data, encoding)
                    
                    if "finish_reason" in data.keys() and not data["finish_reason"] in ["length", "", None]:
                        if "stop_reason" in data.keys() and data["stop_reason"] in stop_strs:
                            chunk_text += data["stop_reason"]
                        flag = 1
                else:
                    data = {}
                    chunk_text = ""
            except:
                print(chunk)
                if "data: " in chunk:
                    data = json.loads(chunk.split("data: ")[1])['error']['message']
                else:
                    data = json.loads(chunk)['message']
                print(data)
                flag = -2
                return flag, data, chunks[-1], last_chunk, index
        else:
            if DEBUG:
                print("="*9)
                print(chunks[0])
            try:
                if "data: " in chunks[0]:
                    data = json.loads(chunks[0].split("data: ")[1])['error']['message']
                else:
                    data = json.loads(chunks[0])['message']
                print(f"{data = }")
                flag = -2
                last_chunk = chunks[0]
            except:
                pass
            return flag, data, chunks[-1], last_chunk, index

    return flag, chunk_text, chunks[-1], last_chunk, index

def response_to_chunk(response_dict, chunksize=100):
    response_str = "data: " + json.dumps(response_dict, ensure_ascii=False) + '\n'*2
    response_chunks = [""]
    for i, s in enumerate(response_str):
        if (i+1) % chunksize == 0:
            response_chunks.append("")
        response_chunks[-1] += s
    if DEBUG:
        print([len(i) for i in response_chunks])
    return response_chunks

def text_to_stream(text_str, this_chunk, template, index=0):
    template_dict = json.loads(template.split("data: ")[1])
    this_chunk_dict = json.loads(this_chunk.split("data: ")[1])

    this_chunk_dict["id"] = template_dict["id"]
    this_chunk_dict["object"] = template_dict["object"]
    this_chunk_dict["created"] = template_dict["created"]
    this_chunk_dict["model"] = template_dict["model"]
    if len(this_chunk_dict["choices"]) != 0:

        if "text" in this_chunk_dict["choices"][0].keys():
            if "text" in template_dict["choices"][0].keys():
                this_chunk_dict["choices"][0]["text"] = text_str
            elif "delta" in template_dict["choices"][0].keys():
                this_chunk_dict["choices"][0] = template_dict["choices"][0]
                this_chunk_dict["choices"][0]["delta"]["content"] = text_str
                this_chunk_dict["choices"][0].pop("text", None)
                this_chunk_dict["choices"][0].pop("stop_reason", None)
        elif "delta" in this_chunk_dict["choices"][0].keys():
            if "content" in this_chunk_dict["choices"][0]["delta"] and this_chunk_dict["choices"][0]["delta"]["content"] != None:
                this_chunk_dict["choices"][0]["delta"]["content"] = text_str
            elif "reasoning_content" in this_chunk_dict["choices"][0]["delta"] and this_chunk_dict["choices"][0]["delta"]["reasoning_content"] != None:
                this_chunk_dict["choices"][0]["delta"]["reasoning_content"] = text_str
        
        this_chunk_dict["choices"][0]["index"] = index
        
    chunks = response_to_chunk(this_chunk_dict)
    return chunks


def get_end_of_string(text, template="</code>"):
    # text = text.strip('\n')
    # for i, t in enumerate(template):
    #     if text[-1] == t:
    #         return template[i+1:]
    # return template
    return ""

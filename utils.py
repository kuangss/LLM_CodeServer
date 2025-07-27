
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

def get_chuncked_content(data_dict):
    if "text" in data_dict.keys():
        return data_dict["text"]
    elif "delta" in data_dict.keys():
        if "content" in data_dict["delta"]:
            return data_dict["delta"]["content"]
    return ""


def concate_chunks(received_text, received_chunks):
    last_chunk = ""
    flag = 0
    chunks = received_chunks.split('\n\n')
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            if chunk != "data: [DONE]":
                last_chunk = chunk
                data = json.loads(chunk.split("data: ")[1])["choices"][0]
                received_text += get_chuncked_content(data)
                if not data["finish_reason"] in ["length", None]:
                    if data["stop_reason"] in ["</answer>"]:
                        # print(data["finish_reason"])
                        flag = 1
            else:
                flag = -1
                if DEBUG:
                    print("="*9)
                    print(chunk)
    return flag, received_text, chunks[-1], last_chunk

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

def text_to_stream(text_str, template):
    template_dict = json.loads(template.split("data: ")[1])

    if "text" in template_dict["choices"][0].keys():
        template_dict["choices"][0]["text"] = text_str
    elif "delta" in template_dict["choices"][0].keys():
        if "content" in template_dict["choices"][0]["delta"]:
            template_dict["choices"][0]["delta"]["content"] = text_str
        
    chunks = response_to_chunk(template_dict)
    return chunks


def get_end_of_string(text, template="</code>"):
    text = text.strip('\n')
    for i, t in enumerate(template):
        if text[-1] == t:
            return template[i+1:]
    return template

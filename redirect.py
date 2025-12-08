from flask import Flask, request, Response
import requests
import re

app = Flask(__name__)

# 配置目标服务器
API_BASE_URL = 'http://api-backend-server:端口'
DEFAULT_BASE_URL = 'http://default-backend-server:端口'

def forward_request(target_url):
    """通用转发函数，适配所有HTTP方法"""
    # 保留原始查询参数
    if request.query_string:
        target_url += f'?{request.query_string.decode()}'
    
    # 构建请求头（移除Host头，添加X-Forwarded-For）
    headers = {key: value for (key, value) in request.headers 
               if key.lower() != 'host'}
    headers['X-Forwarded-For'] = request.remote_addr
    
    # 根据请求方法准备请求参数
    request_kwargs = {
        'method': request.method,
        'url': target_url,
        'headers': headers,
        'cookies': request.cookies,
        'allow_redirects': False,
        'timeout': 30
    }
    
    # 处理不同请求方法的请求体
    if request.method in ['POST', 'PUT', 'PATCH']:
        # 检查是否为文件上传（multipart/form-data）
        if request.content_type and 'multipart/form-data' in request.content_type:
            # 处理文件上传
            files = {}
            data = {}
            
            for key in request.form:
                data[key] = request.form[key]
            
            for key in request.files:
                file_obj = request.files[key]
                files[key] = (file_obj.filename, file_obj.stream, file_obj.content_type)
            
            request_kwargs['files'] = files
            if data:
                request_kwargs['data'] = data
        else:
            # 处理普通请求体（JSON、表单等）
            if request.content_type and 'application/json' in request.content_type:
                # JSON数据
                if request.data:
                    request_kwargs['json'] = request.get_json(silent=True) or {}
            else:
                # 表单数据或其他
                if request.form:
                    request_kwargs['data'] = request.form.to_dict()
                elif request.data:
                    request_kwargs['data'] = request.get_data()
    
    try:
        # 发送请求
        resp = requests.request(**request_kwargs)
        
        # 构建返回给客户端的响应
        excluded_headers = ['content-encoding', 'content-length', 
                           'transfer-encoding', 'connection']
        response_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                           if name.lower() not in excluded_headers]
        
        return Response(resp.content, resp.status_code, response_headers)
    
    except requests.exceptions.ConnectionError:
        return Response('后端服务不可用', status=502)
    except requests.exceptions.Timeout:
        return Response('请求超时', status=504)
    except Exception as e:
        app.logger.error(f'转发请求失败: {str(e)}')
        return Response(f'代理服务器错误: {str(e)}', status=500)

@app.route('/api', defaults={'api_path': ''})
@app.route('/api/<path:api_path>')
def handle_api(api_path):
    """处理所有以/api开头的请求，支持所有HTTP方法"""
    # 自定义API路径修改逻辑
    if api_path:
        # 示例1：版本号升级 (v1 -> v2)
        modified_path = re.sub(r'^v1/', 'v2/', api_path)
        
        # 示例2：添加统一前缀
        # modified_path = f'new-api/{api_path}'
        
        # 示例3：根据请求方法处理路径
        if request.method == 'GET':
            # GET请求特殊处理
            pass
        elif request.method == 'POST':
            # POST请求特殊处理
            pass
        
        target_url = f'{API_BASE_URL}/{modified_path}'
    else:
        # 处理 /api 根路径
        target_url = API_BASE_URL
    
    app.logger.info(f"API请求 {request.method}: /api/{api_path} -> {target_url}")
    return forward_request(target_url)

@app.route('/', defaults={'other_path': ''})
@app.route('/<path:other_path>')
def handle_other(other_path):
    """处理所有非API请求（原样转发），支持所有HTTP方法"""
    if other_path:
        target_url = f'{DEFAULT_BASE_URL}/{other_path}'
    else:
        target_url = DEFAULT_BASE_URL
    
    app.logger.info(f"非API请求 {request.method}: /{other_path} -> {target_url}")
    return forward_request(target_url)

# 添加OPTIONS方法支持（用于CORS预检请求）
@app.route('/api', defaults={'api_path': ''}, methods=['OPTIONS'])
@app.route('/api/<path:api_path>', methods=['OPTIONS'])
@app.route('/', defaults={'other_path': ''}, methods=['OPTIONS'])
@app.route('/<path:other_path>', methods=['OPTIONS'])
def handle_options(api_path='', other_path=''):
    """处理OPTIONS预检请求"""
    # 确定目标URL
    if request.path.startswith('/api'):
        target_base = API_BASE_URL
        path_part = request.path[4:] if len(request.path) > 4 else ''
    else:
        target_base = DEFAULT_BASE_URL
        path_part = request.path[1:] if len(request.path) > 1 else ''
    
    target_url = f'{target_base}/{path_part}' if path_part else target_base
    
    # 转发OPTIONS请求
    return forward_request(target_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith("```python"):
            if last_only:
                program = "" # only extract the last program
            else:
                program += "\n# ========\n"
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return program


# 截断
def process_string(input_str):
    # https://github.com/RUCAIBox/CIR
    # 找到last一个 "```python" 的位置
    code_start_split = input_str.split("```python")
    
    if len(code_start_split) == 1:
        # 如果没有找到 "```python"，直接返回原字符串
        return 0, input_str
    
    # 从 "```python" 之后开始找最近的 "```"
    code_end_split = code_start_split[-1].split('```')

    if len(code_end_split) == 1:
        # 如果没有找到结束的 "```"，直接返回原字符串
        return -1, input_str
    
    # 截取到 "```" 为止（包括 "```"）
    result = "```python".join(code_start_split[:-1]) + "```python" + code_end_split[0] + "```"
    return 1, result

def extract_programs_and_outputs(text: str):
# def extract_programs_and_outputs(text: str) -> list[tuple[str, str]]:
    """
    Extract all Python code blocks and their corresponding output blocks from the text.
    Returns a list of tuples, each tuple contains (program, output).
    If a program has no output block, the output will be an empty string.
    Incomplete or empty blocks are skipped.
    """
    # 新增: 辅助函数用于删除代码块的共同缩进
    def dedent_code(code: str) -> str:
        if not code:
            return code
        
        # 分割成行
        lines = code.splitlines()
        # 找出所有非空行的最小缩进
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # 忽略空行
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            return code
            
        # 对每行删除共同的缩进空白
        dedented_lines = []
        for line in lines:
            if line.strip():  # 非空行
                dedented_lines.append(line[min_indent:])
            else:  # 空行保持原样
                dedented_lines.append(line)
        
        return '\n'.join(dedented_lines)

    results = []
    lines = text.split("\n")
    i = 0
    
    while i < len(lines):
        # Skip until we find a Python code block start
        while i < len(lines) and not lines[i].strip() == "```python":
            i += 1
            
        if i >= len(lines):
            break  # No more Python code blocks
            
        # Start processing Python block
        i += 1  # Skip ```python line
        code_block = ""
        code_complete = False
        
        # Extract code until closing backticks
        while i < len(lines):
            if lines[i].strip() == "```":
                code_complete = True
                i += 1  # Skip closing backticks
                break
            code_block += lines[i] + "\n"
            i += 1
            
        # Skip incomplete or empty code blocks
        if not code_complete or not code_block.strip():
            continue

        # 修改: 在这里对代码块进行dedent处理
        code_block = dedent_code(code_block)
            
        # Now look for an output block
        j = i
        output_block = ""
        output_found = False
        
        # Skip until output block or another Python block
        while j < len(lines):
            if lines[j].strip() == "```output":
                # Found potential output block
                j += 1  # Skip ```output marker
                output_tmp = ""
                output_complete = False
                
                # Extract output until closing backticks
                while j < len(lines):
                    if lines[j].strip() == "```":
                        output_complete = True
                        j += 1  # Skip closing backticks
                        break
                    output_tmp += lines[j] + "\n"
                    j += 1
                    
                if output_complete:
                    output_block = output_tmp
                    output_found = True
                    i = j  # Update main pointer
                    break
                # If incomplete, continue looking
                
            elif lines[j].strip() == "```python":
                # Found another Python block first
                break
                
            j += 1
        
        # Add code-output pair to results
        results.append((code_block, output_block))
    
    return results

def extract_jupyter_like_program(result: str):
    """
    Extract and process programs from text, handling imports and errors appropriately.
    - For programs with errors: keep only import statements
    - For successful programs: remove print statements
    """
    final_program = ""

    programs_and_outputs = extract_programs_and_outputs(result)
    if len(programs_and_outputs) == 0:
        return final_program

    def extract_imports(program: str) -> str:
        """Extract only import statements from program"""
        import_lines = []
        for line in program.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
        return '\n'.join(import_lines) + '\n' if import_lines else ''

    def remove_prints(program: str) -> str:
        """Remove print statements from program"""
        cleaned_lines = []
        for line in program.split('\n'):
            # Skip empty lines
            if not line.strip():
                continue
            # Skip lines that are just print statements
            if line.startswith("print"):
                continue
            if line.startswith("sp.pprint"):
                continue
            if line.startswith("sympy.pprint"):
                continue
            if line.startswith("pprint"):
                continue
            if line.startswith("rprint"):
                continue
            # # Handle print statements that might be part of other code
            # if 'print(' in line:
            #     # If print is not the main statement, keep the line but remove the print
            #     if not line.strip().startswith('print('):
            #         line = line.replace('print(', '# print(')
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines) + '\n' if cleaned_lines else ''

    def is_error_output(output: str) -> bool:
        """Check if output contains error messages"""
        if output is None:
            return False
        error_keywords = [
            'error', 
            'exception', 
            'traceback',
        ]
        output = output.lower()
        return any(keyword in output for keyword in error_keywords)

    def is_import_error_output(output: str) -> bool:
        """Check if output contains error messages"""
        if output is None:
            return False
        error_keywords = [
            'ImportError', 
        ]
        # output = output.lower()
        return any(keyword in output for keyword in error_keywords)

    # Process all programs except the last one
    prev_programs_and_outputs = programs_and_outputs[:-1]
    for program, output in prev_programs_and_outputs:
        if "```python" in program:
            continue
        if (program.strip() != "") and (not is_error_output(output)):
            removed_print_program = remove_prints(program)
            if "print" not in removed_print_program:
                final_program += removed_print_program

    # Process the last program and output
    last_program, last_output = programs_and_outputs[-1]
    if "```python" in last_program:
        last_program_start_pos = last_program.rfind("```python")
        last_program = last_program[last_program_start_pos+len("```python"):]
    final_program += last_program

    return final_program

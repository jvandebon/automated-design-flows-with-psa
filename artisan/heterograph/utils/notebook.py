import sys

def is_notebook():
    return 'ipykernel' in sys.modules

def display_cpp(code):
    from IPython.display import Markdown, display_markdown
    cpp_markdown_template = """```cpp\n%s\n```\n<hr>"""
    code_md = cpp_markdown_template % code
    display_markdown(Markdown(code_md))    


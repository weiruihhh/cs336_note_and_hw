import re

def run_mask_phone_numbers(text: str) -> tuple[str, int]:

    phone_pattern = r'''
            (?<!\d)                                           # 前面不是数字
            (?:\+?1[-.\s]?)?                           # 可选国家代码
            (?:
                \(\d{3}\)[-.\s]?                       # (123) 或 (123)- 或(123)空格 
                |                                       # 或
                \d{3}[-.\s]?                            # 123- 或 123. 或 123空格
            )
            \d{3}[-.\s]?                               # 456- 或 456. 或 456空格
            \d{4}                                       # 7890
            (?:\s?(?:ext|x|extension)\.?\s?\d{2,5})?   # 可选分机号
            (?<!\d)                                            # 后面不是数字
        '''
        
    # 编译正则表达式（使用 VERBOSE 模式以支持注释和格式化）
    compiled_pattern = re.compile(phone_pattern, re.VERBOSE | re.IGNORECASE)
    
    # 执行替换并计数
    masked_text, count = compiled_pattern.subn('|||PHONE_NUMBER|||', text)

    return (masked_text, count)
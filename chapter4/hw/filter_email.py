import re

def run_mask_emails(text: str) -> tuple[str, int]:

    # 业界标准的邮箱正则表达式
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    # 使用 re.subn() 同时完成替换和计数
    masked_text, count = re.subn(
        pattern=email_pattern,
        repl='|||EMAIL_ADDRESS|||',
        string=text
    )
    
    return (masked_text, count)
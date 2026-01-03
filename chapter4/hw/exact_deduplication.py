from collections import Counter

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    #通过计数器的方式得到每个line出现的次数
    lines_count = Counter()
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                lines_count[line] += 1
    #通过集合的方式得到每个line只出现一次的行
    unique_lines = {line for line, count in lines_count.items() if count == 1}

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    for input_path in input_files:
        # 构建输出路径
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_directory, file_name)
        # 打开两个文件
        with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
            # 逐行决策
            for line in infile:
                # 检查是否为唯一行
                if line in unique_lines:
                    # 如果是，写入输出文件
                    outfile.write(line)
    return None

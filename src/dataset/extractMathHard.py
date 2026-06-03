import json
import random

def extract_math500_dataset(input_file, output_file, seed=42):
    # 设置随机种子以保证结果可复现
    if seed is not None:
        random.seed(seed)
        
    level_3_4_data = []
    
    # 1. 读取原始数据并过滤出 level 为 3 和 4 的题目
    print("开始读取并过滤数据集...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get('level') in [3, 4]:
                    level_3_4_data.append(item)
    
    total_filtered = len(level_3_4_data)
    print(f"符合条件（Level 3 或 4）的题目总数: {total_filtered}")
    
    if total_filtered == 0:
        print("未找到符合条件的题目，请检查输入文件。")
        return

    # 2. 计算需要提取的数量：50% 且最多 200 个
    target_count = min(int(total_filtered * 0.5), 200)
    print(f"计算抽样数量 (min({total_filtered} * 50%, 200)): {target_count}")
    
    # 3. 随机打乱并抽取指定数量的样本
    random.shuffle(level_3_4_data)
    extracted_data = level_3_4_data[:target_count]
    
    # 4. 写入新的 jsonl 文件
    print(f"正在保存抽样数据到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("数据集构造完成！")

# 执行提取
input_path = 'hf-math500.jsonl'  # 您的原始文件路径
output_path = 'extracted_math500_level3_4.jsonl'  # 输出文件路径

extract_math500_dataset(input_path, output_path)
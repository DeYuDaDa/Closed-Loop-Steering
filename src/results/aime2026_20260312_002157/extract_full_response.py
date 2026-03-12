import json
from typing import List, Dict, Any

def extract_full_responses(json_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    提取JSON中所有的full_response字段，按顶级分类（Baseline/Continuous/Dynamic_Spherical）分组
    
    参数:
        json_data: 解析后的JSON数据
    返回:
        字典，key为分类名，value为该分类下所有full_response的列表
    """
    # 定义需要提取的顶级分类（和你的JSON结构对应）
    target_categories = ["Baseline", "Continuous", "Dynamic_Spherical"]
    extracted_responses = {cat: [] for cat in target_categories}
    
    for category in target_categories:
        try:
            # 获取该分类下的per_problem列表
            per_problem_list = json_data[category]["per_problem"]
            
            # 遍历每个problem，提取full_response
            for idx, problem in enumerate(per_problem_list):
                try:
                    full_response = problem.get("full_response", "")  # 不存在则返回空字符串
                    extracted_responses[category].append(full_response)
                except Exception as e:
                    print(f"⚠️  处理{category}第{idx}个problem时出错: {e}")
                    extracted_responses[category].append("")  # 出错时填充空字符串，不中断流程
                    
        except KeyError as e:
            print(f"⚠️  {category}下未找到字段: {e}")
        except Exception as e:
            print(f"⚠️  处理{category}时出错: {e}")
    
    return extracted_responses

def save_extracted_responses(extracted_data: Dict[str, List[str]], output_dir: str = "./"):
    """
    将提取的full_response保存为文件（按分类拆分，方便分析）
    """
    # 1. 保存为按分类拆分的TXT文件（易读，适合人工查看/分析）
    for category, responses in extracted_data.items():
        txt_path = f"{output_dir}{category}_full_responses.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for idx, resp in enumerate(responses):
                f.write(f"=== {category} - Problem {idx + 1} ===\n")
                f.write(resp + "\n\n")  # 每个response空行分隔
        print(f"✅ {category}的full_response已保存到: {txt_path}")
    
    # 2. 保存为汇总的JSON文件（适合后续编程分析）
    json_path = f"{output_dir}all_full_responses.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 所有full_response汇总文件已保存到: {json_path}")

def print_response_stats(extracted_data: Dict[str, List[str]]):
    """
    打印提取结果的基础统计信息（快速了解数据情况）
    """
    print("\n=== 提取结果统计 ===")
    total_all = 0
    for category, responses in extracted_data.items():
        total = len(responses)
        total_all += total
        # 统计空值/空白响应数量
        empty_count = sum(1 for resp in responses if not resp.strip())
        # 统计非空响应的平均长度
        non_empty_responses = [resp for resp in responses if resp.strip()]
        avg_length = round(sum(len(resp) for resp in non_empty_responses) / len(non_empty_responses)) if non_empty_responses else 0
        
        print(f"\n{category}:")
        print(f"  - 总数量: {total}")
        print(f"  - 空响应数量: {empty_count}")
        print(f"  - 非空响应平均长度: {avg_length} 字符")
    
    print(f"\n所有分类总计: {total_all} 个full_response")

def main():
    # ========== 配置项（请修改为你的JSON文件路径） ==========
    json_file_path = "experiment_results.json"  # 替换为你的30MB JSON文件路径
    output_directory = "./"  # 提取结果保存的目录（默认当前目录）
    
    # 1. 读取JSON文件（处理大文件，若内存不足可改用流式解析，见下方备注）
    try:
        print("📄 正在读取JSON文件...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        print("✅ JSON文件读取完成")
    
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON解析失败 {e}")
        return
    except MemoryError:
        print("❌ 错误：内存不足！请使用下方的流式解析方案")
        return
    
    # 2. 提取所有full_response
    print("\n🔍 正在提取full_response字段...")
    extracted_responses = extract_full_responses(json_data)
    
    # 3. 打印统计信息
    print_response_stats(extracted_responses)
    
    # 4. 保存提取结果
    print("\n💾 正在保存提取结果...")
    save_extracted_responses(extracted_responses, output_directory)
    
    print("\n🎉 提取完成！你可以查看保存的文件进行后续分析（比如关键词统计、内容分类等）")

# --------------- 备用：内存不足时的流式解析方案 ---------------
def extract_responses_stream(json_file_path: str) -> Dict[str, List[str]]:
    """
    流式提取full_response（适配30MB+超大JSON，避免内存溢出）
    需要先安装ijson：pip install ijson
    """
    import ijson
    extracted_responses = {"Baseline": [], "Continuous": [], "Dynamic_Spherical": []}
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        # 流式解析每个分类下的per_problem元素
        for category in extracted_responses.keys():
            try:
                # 构造路径：Baseline.per_problem.item -> 遍历per_problem的每个元素
                path = f"{category}.per_problem.item"
                for problem in ijson.items(f, path):
                    full_response = problem.get("full_response", "")
                    extracted_responses[category].append(full_response)
                print(f"✅ 流式提取{category}完成")
            except Exception as e:
                print(f"⚠️  流式提取{category}出错: {e}")
    
    return extracted_responses

if __name__ == "__main__":
    # 常规提取（内存足够时用）
    main()
    
    # 若内存不足，注释上面的main()，取消下面的代码：
    # json_file_path = "experiment_results.json"
    # extracted = extract_responses_stream(json_file_path)
    # print_response_stats(extracted)
    # save_extracted_responses(extracted)
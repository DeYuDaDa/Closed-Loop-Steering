import json
import csv
from pathlib import Path
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    加载JSON文件，处理常见的文件读取异常
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        解析后的JSON字典
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：未找到文件 {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"错误：JSON格式解析失败 - {str(e)}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"读取文件时发生未知错误：{str(e)}")

def extract_problem_data(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从JSON数据中提取按组分类的问题数据
    
    Args:
        json_data: 解析后的完整JSON数据
    
    Returns:
        提取后的结构化数据列表，每个元素包含：
        group_name(组名)、id(问题ID)、expected(预期答案)、predicted(预测答案)、correct(是否正确)
    """
    # 定义需要处理的组名
    target_groups = ["Baseline", "Continuous", "Dynamic_Spherical"]
    extracted_data = []
    
    for group_name in target_groups:
        # 检查组是否存在
        if group_name not in json_data:
            print(f"警告：未找到 {group_name} 组数据，跳过该组")
            continue
        
        group_data = json_data[group_name]
        # 检查per_problem字段是否存在且为列表
        if "per_problem" not in group_data or not isinstance(group_data["per_problem"], list):
            print(f"警告：{group_name} 组缺少有效的 per_problem 列表，跳过该组")
            continue
        
        # 遍历该组下的所有问题
        for problem in group_data["per_problem"]:
            # 提取核心字段，处理字段缺失的情况
            problem_id = problem.get("id", "未知ID")
            expected = problem.get("expected", "")
            predicted = problem.get("predicted", "")
            correct = problem.get("correct", None)  # None表示字段缺失
            
            # 整理数据
            extracted_data.append({
                "group_name": group_name,
                "problem_id": problem_id,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "is_correct_str": "正确" if correct is True else "错误" if correct is False else "未知"
            })
    
    return extracted_data

def save_to_csv(data: List[Dict[str, Any]], output_path: str = "problem_answers.csv") -> None:
    """
    将提取的数据保存为CSV文件（方便Excel/表格工具分析）
    
    Args:
        data: 提取后的问题数据
        output_path: 输出CSV文件路径
    """
    # 定义CSV表头
    headers = ["group_name", "problem_id", "expected", "predicted", "correct", "is_correct_str"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n✅ 数据已保存至CSV文件：{output_path}")

def print_analysis_summary(data: List[Dict[str, Any]]) -> None:
    """
    打印数据统计摘要，方便快速排查问题
    """
    print("\n" + "="*80)
    print("📊 数据提取统计摘要")
    print("="*80)
    
    # 按组统计
    groups = {}
    for item in data:
        group = item["group_name"]
        if group not in groups:
            groups[group] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0
            }
        
        groups[group]["total"] += 1
        if item["correct"] is True:
            groups[group]["correct"] += 1
        elif item["correct"] is False:
            groups[group]["incorrect"] += 1
        else:
            groups[group]["unknown"] += 1
    
    # 打印每组的统计
    for group, stats in groups.items():
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"\n📈 {group} 组：")
        print(f"   总问题数：{stats['total']}")
        print(f"   正确数：{stats['correct']}")
        print(f"   错误数：{stats['incorrect']}")
        print(f"   状态未知数：{stats['unknown']}")
        print(f"   准确率：{accuracy:.2f}%")
    
    # 打印字段缺失的问题（重点排查项）
    missing_fields = [item for item in data if item["expected"] == "" or item["predicted"] == "" or item["correct"] is None]
    if missing_fields:
        print(f"\n⚠️  发现 {len(missing_fields)} 个问题存在字段缺失（需重点排查）：")
        for item in missing_fields[:5]:  # 只打印前5个，避免输出过长
            print(f"   组：{item['group_name']} | ID：{item['problem_id']} | 缺失字段：{_get_missing_fields(item)}")
        if len(missing_fields) > 5:
            print(f"   ... 还有 {len(missing_fields)-5} 个缺失字段的问题（详见CSV文件）")

def _get_missing_fields(item: Dict[str, Any]) -> str:
    """辅助函数：获取问题缺失的字段名称"""
    missing = []
    if item["expected"] == "":
        missing.append("expected")
    if item["predicted"] == "":
        missing.append("predicted")
    if item["correct"] is None:
        missing.append("correct")
    return ", ".join(missing)

def main():
    """主函数：执行完整的提取流程"""
    # ===================== 配置项（请根据实际情况修改）=====================
    JSON_FILE_PATH = "/root/Closed-Loop-Steering-System/src/results/MATH500_40_20260316_020927/experiment_results.json"  # 替换为你的JSON文件路径
    OUTPUT_CSV_PATH = "problem_answers_analysis.csv"  # 输出CSV路径
    # =====================================================================
    
    # 1. 加载JSON数据
    print(f"🔍 正在读取JSON文件：{JSON_FILE_PATH}")
    try:
        json_data = load_json_file(JSON_FILE_PATH)
    except Exception as e:
        print(f"❌ 读取JSON文件失败：{e}")
        return
    
    # 2. 提取问题数据
    print("🔧 正在提取问题的expected/predicted/correct字段...")
    extracted_data = extract_problem_data(json_data)
    if not extracted_data:
        print("❌ 未提取到任何问题数据")
        return
    
    # 3. 保存为CSV
    save_to_csv(extracted_data, OUTPUT_CSV_PATH)
    
    # 4. 打印统计摘要
    print_analysis_summary(extracted_data)
    
    print("\n🎉 数据提取完成！你可以：")
    print("   1. 打开CSV文件查看所有问题的详细字段")
    print("   2. 对比expected和predicted的格式差异，分析判别程序的鲁棒性")
    print("   3. 检查correct标记是否符合预期（比如格式小差异是否被误判为错误）")

if __name__ == "__main__":
    main()
import json
from typing import Any, Set

JSON_FILE_PATH = r"C:\Users\m1510\Downloads\experiment_results045merge.json"  # 替换为你的30MB JSON文件路径

def generate_json_schema(
    data: Any,
    path: str = "",
    schema: dict = None,
    visited_paths: Set[str] = None,
    list_depth: int = 1  # 列表只解析前N个元素（通常JSON列表元素结构一致）
) -> dict:
    """
    递归生成JSON的极简结构（仅键名+类型），适配大JSON且适合LLM理解
    
    参数:
        data: 解析后的JSON数据
        path: 当前节点的路径（内部使用）
        schema: 存储结构的字典（内部使用）
        visited_paths: 避免重复解析相同结构的路径（优化大JSON）
        list_depth: 列表元素解析深度（默认1，因为列表元素结构通常一致）
    """
    if schema is None:
        schema = {}
    if visited_paths is None:
        visited_paths = set()
    
    # 处理字典（JSON对象）
    if isinstance(data, dict):
        current_schema = {}
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            # 递归解析子节点，只记录类型
            current_schema[key] = generate_json_schema(
                value, new_path, schema, visited_paths, list_depth
            )
        return current_schema
    
    # 处理列表（JSON数组）- 只解析前list_depth个元素，且只记录结构（不重复）
    elif isinstance(data, list):
        if not data:  # 空列表
            return "list (empty)"
        
        # 提取列表元素的统一结构（避免重复解析大量相同元素）
        element_schemas = []
        for i in range(min(list_depth, len(data))):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            if new_path in visited_paths:
                continue  # 跳过已解析的相同路径结构
            visited_paths.add(new_path)
            element_schema = generate_json_schema(
                data[i], new_path, schema, visited_paths, list_depth
            )
            element_schemas.append(element_schema)
        
        # 列表元素结构统一时，只保留一个示例
        if len(element_schemas) > 0 and all(es == element_schemas[0] for es in element_schemas):
            return f"list (element: {element_schemas[0]})"
        else:
            return f"list (elements: {element_schemas})"
    
    # 基础类型（只返回类型名称，不返回值）
    else:
        return type(data).__name__

def print_simplified_schema(schema: dict, indent: int = 0):
    """
    格式化打印极简结构（适合复制给LLM）
    """
    indent_str = "    " * indent
    for key, value in schema.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}: dict")
            print_simplified_schema(value, indent + 1)
        else:
            print(f"{indent_str}{key}: {value}")

def main():
    # 方式1：解析大JSON文件（推荐，30MB JSON用文件读取更高效）
    json_file_path = JSON_FILE_PATH  # 替换为你的30MB JSON文件路径
    try:
        # 优化大JSON读取：使用utf-8编码，避免编码错误
        with open(json_file_path, "r", encoding="utf-8") as f:
            # 对于超大型JSON，可考虑使用 ijson 流式解析（下方有说明）
            json_data = json.load(f)
        
        # 生成极简结构
        print("=== JSON极简结构（仅键+类型，适合LLM）===")
        schema = generate_json_schema(json_data)
        print_simplified_schema(schema)
        
        # 可选：将结构保存为文件（方便复制给LLM）
        with open("json_schema.txt", "w", encoding="utf-8") as f:
            # 递归写入结构到文件
            def write_schema(schema_dict, file, indent=0):
                indent_str = "    " * indent
                for k, v in schema_dict.items():
                    if isinstance(v, dict):
                        file.write(f"{indent_str}{k}: dict\n")
                        write_schema(v, file, indent + 1)
                    else:
                        file.write(f"{indent_str}{k}: {v}\n")
            write_schema(schema, f)
        print("\n✅ 结构已保存到 json_schema.txt 文件（可直接复制给LLM）")
    
    except FileNotFoundError:
        print(f"❌ 文件不存在：{json_file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误：{e}")
    except MemoryError:
        print("❌ 内存不足！建议使用下方的ijson流式解析方案")

# --------------- 进阶：30MB JSON内存不足时的流式解析方案 ---------------
def stream_json_schema(json_file_path: str):
    """
    流式解析超大JSON（避免一次性加载30MB到内存），仅需安装ijson：pip install ijson
    """
    import ijson
    schema = {}
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            # 流式解析顶层键和类型
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if event == "map_key":  # 捕获键名
                    # 获取该键对应的值类型（下一个event通常是值的类型）
                    next_event = next(parser)
                    val_type = type(next_event[2]).__name__ if next_event[1] != "start_map" else "dict"
                    if next_event[1] == "start_array":
                        val_type = "list"
                    # 构建路径和类型
                    if prefix:
                        path = f"{prefix}.{value}"
                    else:
                        path = value
                    schema[path] = val_type
        
        # 格式化输出流式解析的结构
        print("=== 流式解析的JSON结构 ===")
        for path, typ in schema.items():
            print(f"{path}: {typ}")
    
    except Exception as e:
        print(f"❌ 流式解析失败：{e}")

if __name__ == "__main__":
    main()
    # 如果内存不足，取消注释下方代码使用流式解析：
    # stream_json_schema("your_large_json.json")
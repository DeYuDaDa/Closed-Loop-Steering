import os
import json
from pathlib import Path

def generate_vscode_workspace():
    """
    生成VSCode workspace文件，包含当前目录下的所有文件/文件夹
    """
    # 获取当前目录的绝对路径
    current_dir = Path.cwd().absolute()
    # 定义要排除的目录/文件（可根据自己需求修改）
    exclude_patterns = [
        ".git",          # git版本控制目录
        "node_modules",  # npm依赖目录
        "__pycache__",   # Python缓存目录
        ".venv",         # 虚拟环境
        ".vscode",       # VSCode自身配置目录（可选排除）
        ".code-workspace"# 避免循环包含生成的workspace文件
    ]
    
    # 收集当前目录下的所有文件/文件夹路径
    workspace_folders = []
    for item in os.listdir(current_dir):
        # 跳过排除项
        if item in exclude_patterns:
            continue
        
        # 构造完整路径
        item_path = str(current_dir / item)
        # 添加到workspace配置中（name可选，默认用文件名）
        workspace_folders.append({
            "path": item_path,
            "name": item  # 可选：在VSCode中显示的名称
        })
    
    # 构造VSCode workspace的JSON结构
    workspace_config = {
        "folders": workspace_folders,
        "settings": {}  # 可选：添加自定义设置（如编码、缩进等）
    }
    
    # 生成workspace文件（默认命名为current-workspace.code-workspace）
    workspace_file = current_dir / "current-workspace.code-workspace"
    with open(workspace_file, "w", encoding="utf-8") as f:
        # 格式化JSON，便于阅读
        json.dump(workspace_config, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Workspace文件已生成：{workspace_file}")
    print(f"📂 共添加 {len(workspace_folders)} 个文件/文件夹")
    
    # 可选：自动用VSCode打开这个workspace（需确保code命令已配置）
    try:
        os.system(f"code {workspace_file}")
        print("🚀 已自动在VSCode中打开该workspace")
    except Exception as e:
        print(f"⚠️  自动打开失败（需配置VSCode的code命令）：{e}")
        print("💡 手动打开方式：在VSCode中 → 文件 → 将文件夹添加到工作区 → 选择生成的.code-workspace文件")

if __name__ == "__main__":
    generate_vscode_workspace()
import json
import re
import sympy
import os
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def normalize_math(s):
    """归一化数学表达式字符串"""
    if s is None: return str(s)
    s = str(s).strip()
    # 移除 LaTeX 干扰项
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mbox\{([^}]*)\}', r'\1', s)
    s = s.replace(',', '').replace(r'\!', '').replace('$', '')
    s = re.sub(r'\^\\circ\s*', '', s)
    s = s.replace('^{\\circ}', '').replace('^\circ', '')
    s = s.replace(r'\dfrac', r'\frac').replace(r'\left', '').replace(r'\right', '')
    
    # 移除变量前缀
    if s.startswith(('x=', 'y=', 'n=')): 
        s = s[2:].strip()
    
    # 分数格式转换
    s = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\s*(\d)\s*(\d)', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\s*(\d)\s*\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\s*\{([^{}]+)\}\s*(\d)', r'(\1)/(\2)', s)
    
    s = s.replace(' ', '').replace('cents', '').replace('units', '')
    return s

def check_equiv(exp, pred):
    """检查预期与预测是否等价"""
    if pred is None or pred == "": return False
    e = normalize_math(exp)
    p = normalize_math(pred)
    if e == p: return True
    
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        e_val = parse_expr(e, transformations=transformations)
        p_val = parse_expr(p, transformations=transformations)
        if e_val.equals(p_val): return True
        if abs(float(e_val) - float(p_val)) < 1e-6: return True
    except:
        pass
        
    # 处理坐标/元组
    e_tuple = e.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
    p_tuple = p.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
    if len(e_tuple) == len(p_tuple) and len(e_tuple) > 1:
        match = True
        for ei, pi in zip(e_tuple, p_tuple):
            try:
                if not parse_expr(ei, transformations=transformations).equals(parse_expr(pi, transformations=transformations)):
                    match = False; break
            except:
                if ei != pi: match = False; break
        if match: return True
    return False

def fix_and_overwrite(file_path='result.json'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"正在读取大文件 {file_path} (请稍候)...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("正在进行鲁棒性修复与重新统计...")
    for group_name, group_stats in data.items():
        if not isinstance(group_stats, dict) or 'per_problem' not in group_stats:
            continue
            
        problems = group_stats['per_problem']
        fixed_this_group = 0
        
        for problem in problems:
            # 仅修复原判为 False 且预测值不为空的条目
            if not problem.get('correct', False) and problem.get('predicted'):
                if check_equiv(problem.get('expected'), problem.get('predicted')):
                    problem['correct'] = True
                    fixed_this_group += 1
        
        # 重新计算该组指标
        total = len(problems)
        correct_count = sum(1 for p in problems if p.get('correct', False))
        accuracy = correct_count / total if total > 0 else 0
        
        group_stats['correct_count'] = correct_count
        group_stats['accuracy'] = accuracy
        print(f"[{group_name}] 修复完成: 修正数 {fixed_this_group}, 新准确率 {accuracy:.2%}")

    print(f"正在覆盖原文件 {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        # 为了压缩空间，这里不使用 indent (或者使用 indent=2 如果你需要可读性)
        # ensure_ascii=False 可以减少由于转义导致的体积膨胀
        json.dump(data, f, ensure_ascii=False)
        
    print("✅ 修复完毕，原文件已更新。")

if __name__ == "__main__":
    fix_and_overwrite('result.json')
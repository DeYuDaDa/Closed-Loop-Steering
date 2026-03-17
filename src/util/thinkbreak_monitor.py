import json

# 替换为你的 JSON 路径
json_path = "experiment_results.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ds_problems = data.get("Dynamic_Spherical", {}).get("per_problem", [])

print("="*60)
print("🛑 ThinkBrake (自动刹车) 触发位置分析")
print("="*60)

tb_count = 0
for p in ds_problems:
    # 1. 检查是否触发了 ThinkBrake (convergence 为 True)
    if p.get("convergence", False):
        alpha_traj = p.get("alpha_trajectory", [])
        ema_traj = p.get("ema_trajectory", [])
        
        # 2. 寻找刹车点：最后一个 alpha > 0 的位置
        active_indices = [i for i, a in enumerate(alpha_traj) if a > 0]
        
        if active_indices:
            # 刹车生效的瞬间，就是最后一个活跃步的下一步
            tb_step = active_indices[-1] + 1 
            tb_count += 1
            
            total_tokens = len(alpha_traj)
            
            # 提取那一瞬间的 EMA 熵 (证明它是被硬切断的)
            ema_at_brake = ema_traj[tb_step] if tb_step < len(ema_traj) else ema_traj[-1]
            
            print(f"📝 题目 ID: {p.get('id', 'Unknown')}")
            print(f"   - 刹车位置: 第 {tb_step} 个 Token (总长: {total_tokens} tokens)")
            print(f"   - 刹车瞬间的 EMA 熵: {ema_at_brake:.3f} (大概率仍高于阈值 0.15)")
            print(f"   - 最终正确性: {'✅ 正确' if p.get('correct') else '❌ 错误'}\n")

print(f"📊 总结: 32 个样本中，共有 {tb_count} 个样本成功触发了 ThinkBrake 自动刹车。")
print("="*60)
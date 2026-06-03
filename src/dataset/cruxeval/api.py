import json
import os
import asyncio
import platform
import random
import time
from openai import AsyncOpenAI

# 导入 CRUXEval 提示词模板
from prompts import (
    make_direct_output_prompt,
    make_cot_output_prompt,
    make_direct_input_prompt,
    make_cot_input_prompt,
)

# 异步文件写入锁，确保高并发时安全写入
file_lock = asyncio.Lock()

class RateLimiter:
    """速率限制器，确保每分钟请求数控制在设定值以内 (RPM)"""
    def __init__(self, requests_per_minute: int):
        self.delay = 60.0 / requests_per_minute
        self.lock = asyncio.Lock()
        self.last_call = 0.0

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self.last_call = time.monotonic()

def clean_content(content: str) -> str:
    """清理模型的 Markdown 代码块等格式"""
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1]
            if content.startswith("python"):
                content = content[6:]
            elif content.startswith("py"):
                content = content[2:]
    return content.strip()

def extract_prediction_output(content: str) -> str:
    """从输出预测回复中提取最后的有效值"""
    if "[/ANSWER]" in content:
        content = content.split("[/ANSWER]")[0]
    elif "[ANSWER]" in content:
        content = content.split("[ANSWER]")[1]
    
    if "==" in content:
        content = content.split("==")[1]
    return content.strip()

def extract_prediction_input(content: str) -> str:
    """从输入预测回复中提取输入形式，格式化为 f(args)"""
    if "[/ANSWER]" in content:
        content = content.split("[/ANSWER]")[0]
    elif "[ANSWER]" in content:
        content = content.split("[ANSWER]")[1]
    
    if "==" in content:
        content = content.split("==")[0].strip()
    if "assert f" in content:
        content = "f" + content.split("assert f")[1].strip()
    else:
        content = content.strip()
        if not content.startswith("f("):
            content = f"f({content})"
    return content.strip()

def check_correctness(code: str, gold_output: str, prediction: str, mode: str) -> bool:
    """通过本地执行代码和断言检查预测是否正确"""
    try:
        if mode == "output":
            extracted = extract_prediction_output(prediction)
            code_to_execute = f"assert {gold_output} == {extracted}"
        else:  # input
            extracted = extract_prediction_input(prediction)
            code_to_execute = f"{code}\nassert {gold_output} == {extracted}"
            
        exec_globals = {}
        exec(code_to_execute, exec_globals)
        return True
    except Exception:
        return False

async def write_result(output_file: str, result_dict: dict):
    """加锁安全写入文件，每处理完一条实时写入一条"""
    async with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

async def process_task(item_data: dict, index: int, client: AsyncOpenAI, rate_limiter: RateLimiter, semaphore: asyncio.Semaphore, output_file: str, mode: str, cot: bool):
    """处理单条推理请求"""
    await rate_limiter.wait()
    async with semaphore:
        custom_id = item_data.get("id", f"sample_{index}")
        try:
            code = item_data.get("code", "")
            input_val = item_data.get("input", "")
            output_val = item_data.get("output", "")
            
            if mode == "output":
                prompt_fn = make_cot_output_prompt if cot else make_direct_output_prompt
                input_query = (code, input_val)
            else:  # input
                prompt_fn = make_cot_input_prompt if cot else make_direct_input_prompt
                input_query = (code, output_val)
                
            input_text = prompt_fn(input_query)
            messages = [
                {"role": "user", "content": input_text}
            ]
            model = "qwen3-8b" 
        except Exception as e:
            await write_result(output_file, {
                "id": custom_id,
                "status": "error",
                "error_message": f"数据解析/Prompt构建失败: {str(e)}",
                "index": index
            })
            return

        try:
            response_stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                top_p=0.95,
                stream=True,
                extra_body={
                    "top_k": 20,
                    "enable_thinking": True,
                    "thinking_budget": 32768
                }
            )
            
            full_reasoning_content = ""
            full_content = ""
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    full_reasoning_content += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    full_content += delta.content
            
            result = {
                "id": custom_id,
                "status": "success",
                "code": code,
                "input": input_val,
                "output": output_val,
                "reasoning_content": full_reasoning_content,
                "content": full_content
            }
            
        except Exception as e:
            result = {
                "id": custom_id,
                "status": "error",
                "error_message": str(e)
            }

        await write_result(output_file, result)

async def main(input_file: str, mode: str = "output", cot: bool = False, sample_size: int = 100):
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_{mode}_{'cot_' if cot else ''}sample_result.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    
    api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-1ed0e250e8d74b17beda91e1b889bc96")
    if not api_key:
        raise ValueError("未读取到 DASHSCOPE_API_KEY 环境变量，请先配置。")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 限制并发与请求速率 (30 RPM = 每2秒发送一个请求)
    rate_limiter = RateLimiter(requests_per_minute=30)
    semaphore = asyncio.Semaphore(15) # 同时进行的并发请求数上限
    tasks = []
    
    print(f"正在读取CRUXEval JSONL数据集：{input_file}")
    examples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                
    total_count = len(examples)
    if total_count == 0:
        print("错误: 未在文件中找到任何有效数据样本。")
        return
        
    # 随机抽取指定数量样本
    actual_sample_size = min(sample_size, total_count)
    sampled_examples = random.sample(examples, actual_sample_size)
    
    print(f"数据集总数: {total_count} 条，已随机抽取 {actual_sample_size} 条进行测试。")
    print(f"已开启并发限速：30 RPM (每分钟最大30个请求)，处理中，请稍候...")
    
    for index, item in enumerate(sampled_examples):
        task = asyncio.create_task(process_task(item, index, client, rate_limiter, semaphore, output_file, mode, cot))
        tasks.append(task)
            
    await asyncio.gather(*tasks)
    print(f"🎉 全部任务处理完毕！最终结果已保存至：{output_file}")
    
    # ==========================================
    # 实时读取刚才生成的 jsonl 并计算正确率
    # ==========================================
    print("\n📊 正在统计本次测试的正确率...")
    correct_count = 0
    valid_tasks = 0
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res = json.loads(line)
            if res.get("status") == "success":
                code = res.get("code", "")
                input_val = res.get("input", "")
                output_val = res.get("output", "")
                content = clean_content(res.get("content", ""))
                
                is_correct = check_correctness(code, output_val, content, mode)
                
                valid_tasks += 1
                if is_correct:
                    correct_count += 1
                
                print(f"样本 {res.get('id')}: {'✅ 正确' if is_correct else '❌ 错误'}")
                    
    if valid_tasks > 0:
        accuracy = correct_count / valid_tasks
        print("-" * 40)
        print(f"成功完成请求的样本数: {valid_tasks}")
        print(f"模型预测正确样本数: {correct_count}")
        print(f"本次测试最终正确率 (Accuracy): {accuracy:.2%}")
        print("-" * 40)
    else:
        print("❌ 未能成功统计正确率，可能所有请求均响应失败。")

if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    INPUT_FILENAME = os.path.join("data", "cruxeval.jsonl")
    if not os.path.exists(INPUT_FILENAME):
        print(f"错误: 找不到输入文件 {INPUT_FILENAME}")
    else:
        # 可以通过调整 mode ("output" 或 "input"), cot (True 或 False) 自定义评估
        asyncio.run(main(INPUT_FILENAME, mode="output", cot=False, sample_size=100))
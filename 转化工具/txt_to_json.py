"""
纯文本 (.txt) → JSON 题目转换工具。

通过 DeepSeek AI 自动识别并拆分 TXT 文件中的数学题目，
不再依赖正则表达式规则切割，相比固定规则能更准确地识别
各种格式的题目边界和领域分类。

用法:
    python 转化工具/txt_to_json.py <txt路径> [-o 输出.json] [--max N]

输出 JSON 格式:
[
  {
    "id": "代数_001",
    "question": "题目完整内容",
    "domain": "代数",
    "reference_answer": ""
  }
]
"""
import argparse
import asyncio
import json
import os
import sys

from openai import OpenAI


def _read_file(txt_path: str) -> str:
    """读取 TXT 文件，尝试多种编码"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312", "utf-16"]:
        try:
            with open(txt_path, "r", encoding=encoding) as f:
                content = f.read()
            if content and content[0] == "\ufeff":
                content = content[1:]
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法读取文件，编码不支持: {txt_path}")


def _load_deepseek_key() -> str | None:
    """从 .env 或环境变量中读取 DeepSeek API Key"""
    # 先尝试环境变量
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key and not key.startswith("your_"):
        return key

    # 从 .env 文件加载
    env_paths = [
        os.path.join(os.path.expanduser("~"), ".math_evaluator", ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "测试工具", ".env"),
    ]
    for env_path in env_paths:
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEEPSEEK_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _create_client() -> OpenAI:
    """创建 DeepSeek OpenAI 兼容客户端"""
    api_key = _load_deepseek_key()
    if not api_key:
        raise RuntimeError(
            "未找到 DeepSeek API Key，请先在设置中配置。\n"
            "配置文件: ~/.math_evaluator/.env"
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


SPLIT_PROMPT = """你是一个数学题目解析助手。请将以下文本中的数学题目拆分为独立的题目列表。

要求：
1. 识别并拆分每道独立的数学题
2. 为每道题提取完整的问题内容（包括所有小问）
3. 判断每道题的知识领域（如：代数、几何、函数、概率统计、数列、微积分等）
4. 如果有子领域信息（如"代数·一元二次方程"），也请包含

输出格式必须是纯 JSON 数组：
[
  {
    "id": "领域_序号",
    "question": "完整题目内容（包含所有小问和多行文本）",
    "domain": "主要领域"
  }
]

注意：
- 只返回 JSON 数组，不要有其他文字
- 每道题的 question 字段必须完整，不能截断
- 领域名称使用中文

====================
文本内容：
====================
"""


def convert_txt(txt_path: str, max_problems: int = 0) -> list[dict]:
    """
    使用 DeepSeek AI 将 TXT 文件转换为题目列表。

    通过 AI 识别题目边界和领域分类，相比正则切割更准确。

    参数:
        txt_path: TXT 文件路径
        max_problems: 最大提取题目数（0 表示全部，截断在 AI 返回后执行）

    返回:
        题目字典列表，每项含 id/question/domain/reference_answer
    """
    content = _read_file(txt_path)

    if not content.strip():
        raise ValueError("文件内容为空")

    # 限制输入长度，防止超出 token 限制
    max_input_chars = 12000
    if len(content) > max_input_chars:
        print(f"提示: 文件较大 ({len(content)} 字符)，截取前 {max_input_chars} 字符进行识别")
        content = content[:max_input_chars]

    client = _create_client()

    print("正在调用 DeepSeek AI 识别并拆分题目...")

    try:
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": SPLIT_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        result_text = response.choices[0].message.content.strip()

        # 提取 JSON 数组（处理 markdown 代码块包裹的情况）
        if result_text.startswith("```"):
            # 找到第一个 ``` 之后和最后一个 ``` 之前的内容
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_text)
            if match:
                result_text = match.group(1).strip()

        problems = json.loads(result_text)

        if not isinstance(problems, list):
            raise ValueError("AI 返回的不是题目数组")

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"AI 返回的内容无法解析为 JSON:\n{str(e)}\n\n"
            f"AI 原始返回:\n{result_text[:500]}..."
        )
    except Exception as e:
        raise RuntimeError(f"AI 识别失败: {str(e)}")

    # 确保每道题都有必要字段
    domain_counters: dict[str, int] = {}
    for problem in problems:
        if "question" not in problem or not problem["question"]:
            continue
        domain = problem.get("domain", "未知")
        # ID 取领域主类（· 之前），避免特殊字符
        domain_key = domain.split("·")[0].strip().replace(" ", "_") if "·" in domain else domain
        if domain_key not in domain_counters:
            domain_counters[domain_key] = 0
        domain_counters[domain_key] += 1
        problem["id"] = f"{domain_key}_{domain_counters[domain_key]:03d}"
        if "reference_answer" not in problem:
            problem["reference_answer"] = ""

    print(f"AI 识别出 {len(problems)} 道题目")

    if max_problems > 0 and len(problems) > max_problems:
        problems = problems[:max_problems]
        print(f"限制为前 {max_problems} 道")

    return problems


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(description="TXT → JSON 题目转换器（AI 智能拆分）")
    parser.add_argument("txt", help="TXT 文件路径 (.txt)")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--max", type=int, default=0, help="最多提取题目数 (0=全部)")
    args = parser.parse_args()

    txt_path = args.txt
    if not os.path.exists(txt_path):
        print(f"文件不存在: {txt_path}")
        sys.exit(1)

    problems = convert_txt(txt_path, max_problems=args.max)

    if not problems:
        print("警告: 未解析出任何题目，请检查文件格式。")
        sys.exit(1)

    # 默认输出路径
    if args.output is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "测试结果", "原始问题")
        os.makedirs(output_dir, exist_ok=True)
        txt_name = os.path.splitext(os.path.basename(txt_path))[0]
        args.output = os.path.join(output_dir, f"{txt_name}.json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {args.output}")
    print(f"共 {len(problems)} 道题目")
    for p in problems:
        print(f"  - {p['id']}: {p.get('domain', '?')}")


if __name__ == "__main__":
    main()

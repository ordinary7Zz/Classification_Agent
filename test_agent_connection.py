"""
简单的测试脚本，用于测试与百炼（DashScope）大模型的连接是否成功
读取 config/config.yaml 中的 agent_llm 配置（api_key、model_name），
通过 OpenAI 兼容接口发送一条测试消息，并打印返回结果。
"""

import sys
from pathlib import Path

import yaml
from openai import OpenAI


def load_config(config_path: str = "config/config-TN5K.yaml"):
    """加载配置文件（使用 yaml 解析）"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_agent_connection(api_key: str, model_name: str, max_retries: int = 3) -> bool:
    """
    测试与百炼大模型的连接

    Args:
        api_key: 百炼（DashScope）API Key
        model_name: 模型名称（如 qwen3.5-flash）
        max_retries: 最大重试次数
    """
    print("=" * 70)
    print("测试百炼（DashScope）大模型连接")
    print("=" * 70)

    if not api_key:
        print("✗ 未提供 API Key")
        return False

    masked_key = api_key if len(api_key) <= 16 else f"{api_key[:8]}...{api_key[-4:]}"
    print(f"\nAPI Key: {masked_key}")
    print(f"模型名称: {model_name}")
    print(f"最大重试次数: {max_retries}\n")

    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    test_message = "你好，请只回复：连接成功。"
    print(f"发送测试消息: {test_message}")
    print("\n等待响应...\n")

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"\n重试第 {attempt} 次...\n")

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_message}],
            )
            choice = completion.choices[0]
            content = (getattr(choice, "message", None) or choice).content
            content = content.strip() if isinstance(content, str) else str(content)

            print("模型返回内容：")
            print(content)

            print("\n" + "=" * 70)
            print("✓ 请求发送成功，已收到模型响应")
            print("=" * 70)

            if "连接成功" in content:
                print("\n检测到关键字“连接成功”，判定连接测试通过。")
            else:
                print("\n未检测到关键字“连接成功”，但模型已正常返回结果。")

            return True

        except Exception as e:
            last_error = e
            err_type = type(e).__name__
            print(f"\n尝试 {attempt}/{max_retries} 失败: {err_type}: {e}")

    print("\n" + "=" * 70)
    print("✗ 连接失败")
    print("=" * 70)

    if last_error is not None:
        print(f"\n最后一次错误类型: {type(last_error).__name__}")
        print(f"错误信息: {last_error}")

    print("\n可能的原因：")
    print("1. API Key 无效或已过期（请在百炼控制台检查）")
    print("2. 模型名称不正确或未开通（如 qwen3.5-flash）")
    print("3. 网络问题（本机无法访问 dashscope.aliyuncs.com）")
    print("4. 账户额度不足或被限制")

    return False


def main() -> int:
    """主函数"""
    try:
        print(">>> 加载配置文件...")
        config = load_config()
        agent_llm_cfg = config.get("agent_llm", {}) or {}

        api_key = agent_llm_cfg.get("api_key", "") or ""
        model_name = agent_llm_cfg.get("model_name", "qwen3.5-flash")

        if not api_key:
            print("⚠️ 配置中的 agent_llm.api_key 为空，将尝试从环境变量 DASHSCOPE_API_KEY 读取。")
            from os import getenv

            api_key = getenv("DASHSCOPE_API_KEY", "")

        if not api_key:
            print("✗ 未能从配置或环境变量中获取 API Key。")
            return 1

        print("✓ 配置加载成功\n")

        success = test_agent_connection(api_key, model_name, max_retries=3)

        if success:
            print("\n" + "=" * 70)
            print("测试完成：连接正常")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("测试完成：连接失败，请检查配置或网络")
            print("=" * 70)
            return 1

    except FileNotFoundError as e:
        print(f"\n✗ 错误: {e}")
        print("请确保 config/config.yaml 文件存在")
        return 1
    except Exception as e:
        print(f"\n✗ 未预期的错误: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


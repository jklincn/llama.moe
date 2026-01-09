import openai

try:
    client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-1234")

    # 自动查询模型
    models_list = client.models.list()
    if not models_list.data:
        raise ValueError("未找到任何可用模型")
    
    model_id = models_list.data[0].id
    print(f"自动检测到模型: {model_id}")

    messages = [
        {"role": "user", "content": "你好！请你用中文介绍一下自己。"},
    ]

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.7,
    )

    response_content = completion.choices[0].message.content
    print(f"模型回复: {response_content}\n")

except openai.APIConnectionError as e:
    print("连接错误: 无法连接到服务。")
    print(
        "请确保你的 `llama-server` 正在运行，并且 `base_url` 配置正确。"
    )
    print(f"详细错误: {e.__cause__}")
except Exception as e:
    print(f"发生了一个未预料到的错误: {e}")

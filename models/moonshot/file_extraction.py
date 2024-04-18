from pathlib import Path
from openai import OpenAI
import os

api_key = os.getenv("MOONSHOT_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

# xlnet.pdf 是一个示例文件, 我们支持 pdf, doc 以及图片等格式, 对于图片和 pdf 文件，提供 ocr 相关能力
file_object = client.files.create(file=Path("moonshot/xlnet.pdf"), purpose="file-extract")

# 获取结果
# file_content = client.files.retrieve_content(file_id=file_object.id)
# 注意，之前 retrieve_content api 在最新版本标记了 warning, 可以用下面这行代替
# 如果是旧版本，可以用 retrieve_content
file_content = client.files.content(file_id=file_object.id).text

# 把它放进请求中
question = "根据 xlnet.pdf，给出调用其推理服务的 python 代码示例，请求数据是一张名为 test_det.jpg 的图片，然后将返回数据解析并将 bboxes 中的左上和右下坐标，在原图上画出红色的矩形框，并将图片保存为 resp_det.jpg"

messages=[
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
    },
    {
        "role": "system",
        "content": file_content,
    },
    {"role": "user", "content": question},
]

print(question)
print("-----------------------------------------------------")

# 然后调用 chat-completion, 获取 kimi 的回答
completion = client.chat.completions.create(
  model="moonshot-v1-32k",
  messages=messages,
  temperature=0.3,
)

print(completion.choices[0].message.content)
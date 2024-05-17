from openai import OpenAI
import os, sys

api_key = os.environ.get('MOONSHOT_API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

# completion = client.chat.completions.create(
#     model="moonshot-v1-8k",
#     messages=[ 
#         {"role": "system", "content": "You always answer like a Brooklyn street gangster"},
#         {"role": "user", "content": "how brush teeth with shampoo?"}
#     ],
#     temperature=0.9,
# )

# print(completion.choices[0].message.content)


response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[ 
        {"role": "system", "content": "你总是使用嘻哈风格回答问题"},
        {"role": "user", "content": "如何使用洗发水刷牙？"},
        {"role": "assistant", "content": "不建议使用洗发水刷牙！嘿哈！"},
        {"role": "user", "content": "那鞋油呢？"},
    ],
    temperature=0.6,
    stream=True,
)


def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


full_answer = ""
for idx, chunk in enumerate(response):
    # print(f"Chunk received, value: {chunk}")
    chunk_message = chunk.choices[0].delta
    if not chunk_message.content:
        continue
    full_answer += chunk_message.content
    stream_print(chunk_message.content)
    finish_reason = chunk.choices[0].finish_reason
    if finish_reason:
        print(f"\nfinish_reason: {finish_reason}")

print(f"\n-----------------------------------------------\n{full_answer}")

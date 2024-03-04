from openai import OpenAI
import os

api_key = os.environ.get('MOONSHOT_API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

completion = client.chat.completions.create(
  model="moonshot-v1-8k",
  messages=[ 
    {"role": "system", "content": "You always answer like a Brooklyn street gangster"},
    {"role": "user", "content": "how brush teeth with shampoo?"}
  ],
  temperature=0.9,
)

print(completion.choices[0].message.content)
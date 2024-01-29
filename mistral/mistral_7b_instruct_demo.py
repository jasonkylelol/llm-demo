from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

device = "cuda:1" # the device to load the model onto
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

model_path = "/root/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_path,
    device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path,
    device_map=device, use_fast=True)
pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=True,
    return_full_text=False,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=1.1,
    pad_token_id=2,
)

# messages = [
#     # {"role": "system", "content": "you always response like a thug"},
#     {"role": "user", "content": "how brush teeth with shampoo?"},
#     {"role": "assistant", "content": "do not even try it to put shampoo to your teeth, you hear me? shampoo is designed to use for hair, not oral!"},
#     {"role": "user", "content": "how about mercury?"}
# ]

question = 'who is Joshua Davis and what happened to him?'

input_template = f'''you always response with pure JSON blob with key: "input" with value "{question}", and put your answer to the value of key: "AI"'''

messages = [
    {
        "role": "user",
        "content": input_template,
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("=========================================")

outputs = pipe(text_inputs=prompt)
print(outputs[0]["generated_text"])

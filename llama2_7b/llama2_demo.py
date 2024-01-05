import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/root/huggingface/models/meta-llama/Llama-2-7b-chat-hf'

max_new_tokens=256
top_k=50
top_p=0.65
temperature=0.95


model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:2",
    load_in_4bit=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=True,
    return_full_text=False,
    # max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=1.1
)


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = [{"role": "system", "content": "You always answer like a gangster"}] # not working
    for item in history_transformer_format:
        messages.append({"role": "user", "content": item[0]})
        if item[1] != "":
            messages.append({"role": "assistant", "content": item[1]})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt, "\n==================================")

    outputs = pipe(text_inputs=prompt)
    print(outputs[0]["generated_text"])

    # model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:1")
    # output = model.generate(**model_inputs)
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)


if __name__ == '__main__':
    history = [
        (
            'How brush teeth with shampoo?',
            'Brushing teeth with shampoo is not recommended as it can cause oral health problems'
        ),
    ]
    msg = 'Is mercury an option?'
    predict(msg, history)

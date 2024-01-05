import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://192.168.2.75:8080")

'''
 <|user|>
how brush teeth with shampoo</s>
<|assistant|>
'''

def inference(message, history):
    history_transformer_format = history + [[message, ""]]
    #messages = []
    prompt = ""
    for item in history_transformer_format:
        prompt += "<|user|>\n" + item[0] + "</s>\n"
        if item[1] != "":
            prompt += "<|assistant|>" + item[1] + "\n"
        else:
            prompt += "<|assistant|>"
        #messages.append({"role": "user", "content": item[0]})
        #if item[1] != "":
        #    messages.append({"role": "assistant", "content": item[1]})

    #prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt, "\n==================================")

    partial_message = ""
    for token in client.text_generation(prompt, max_new_tokens=256, stream=True, do_sample=True, top_p=0.05, top_k=50, temperature=0.7):
        partial_message += token
        yield partial_message

if __name__ == '__main__':
    gr.ChatInterface(
        inference,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
        description="HuggingFaceH4/zephyr-7b-beta demo for Gradio UI consuming TGI endpoint with LLM",
        title="Gradio 🤝 TGI",
        examples=["Are tomatoes vegetables?"],
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
    ).queue().launch(server_name='0.0.0.0')


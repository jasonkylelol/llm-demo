from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
import uvicorn, json, asyncio
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate

app = FastAPI()

ernie_bot_llm = QianfanChatEndpoint()

llm_map = {
    "ernie-bot": ernie_bot_llm,
}

# curl -X POST -N -d '{"dialog":[{"role":"user","content":"How brush teeth with shampoo?"},{"role":"ai","content":"Brushing teeth with shampoo is not recommended as it can cause oral health problems"},{"role":"user","content":"Is mercury an option?"}]}' http://127.0.0.1:8060/thirdpart_llm?llm_type=ernie-bot

def get_llm_by_type(llm_type: str):
    return llm_map.get(llm_type)

def init_prompts(request_body: dict):
    messages = []
    dialog_items = request_body.get("dialog")
    for dialog in dialog_items:
        role = dialog.get("role")
        content = dialog.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
        else:
            print(f"unsupport role:{role}")

    prompts = ChatPromptTemplate.from_messages(messages).format_messages()
    return prompts

def call_llm(request_body: dict, llm_type: str) -> str :
    prompts = init_prompts(request_body)
    llm = get_llm_by_type(llm_type)
    response = llm.invoke(prompts)
    return response.content

@app.post("/thirdpart_llm")
async def thirdpart_llm(
    request: Request,
    llm_type: str = Query(..., title="LLM Type", description="Type of Large Language Model"),
):
    # Now you can use llm_type in your function
        # For example, print it for now
    print(f"Received LLM Type: {llm_type}")

    # Get the raw content from the request body
    raw_body = await request.body()
    # Parse the raw content as JSON
    request_body = json.loads(raw_body)
    if not request_body:
        raise HTTPException(status_code=400, detail="invalid request body")
    
    try:
        response_body = call_llm(request_body, llm_type)

        # Convert the bytes to an iterable (generator)
        async def generate():
            chunks = response_body.split(' ')
            for chunk in chunks:
                # print(chunk)
                yield chunk + ' '
                # Introduce a delay between chunks (adjust the sleep duration as needed)
                await asyncio.sleep(0.01)

        # Return a StreamingResponse using the generator function
        return StreamingResponse(content=generate(), media_type="text/event-stream")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8060)
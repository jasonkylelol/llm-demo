import bs4, time, os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_demo.custom.text_splitter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from langchain_community.document_loaders.text import TextLoader

os.environ["USER_AGENT"] = "DefaultLangchainUserAgent"

api_base_url = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")
llm_model = os.getenv("LLM_MODEL") or "glm-4-flash"
embedding_model = os.getenv("EMBEDDING_MODEL") or "embedding-3"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0.1,
    base_url=api_base_url,
    api_key=api_key,
)

embedding = OpenAIEmbeddings(
    model=embedding_model,
    base_url=api_base_url,
    api_key=api_key,
    check_embedding_ctx_length=False,
)

### Construct retriever ###
# loader = TextLoader("cache/files/opanai_lawsuit.txt")
loader = TextLoader("cache/files/xiaomi.txt")
docs = loader.load()

text_splitter = ChineseRecursiveTextSplitter(chunk_size=500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

print(f"vectorstore by Chroma")

### Contextualize question ###

# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )
contextualize_q_system_prompt = (
    "给定一个聊天记录和与聊天记录有关的用户问题，重新生成一个独立的问题，"
    "该问题囊括了聊天记录的关键信息，使得该问题无需聊天记录即可理解。"
    "不要回答此问题，只需重新表述此问题，如果无法表述就按原样返回。"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n##################################################\n\n"
#     "{context}"
# )
system_prompt = (
    "你是擅长回答问题的智能助手，使用提供的已知信息来回答问题。"
    "如果你不知道答案，就说不知道。请保持答案简洁和准确。\n以下是已知信息:\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# questions = [
#     "What is Mr.Musk seeking to do with OpenAI in this lawsuit?",
#     "What would you do if you were Mr. Musk?",
#     "in my opinion, require OpenAI to open up its technology to others, which is much more valuable than repay him the money he donated. what do you think?",
# ]
questions = [
    "小米汽车都有哪些车型？价格分别是多少？",
    "预算25万可以选择什么？",
    "此版本的续航是多少？",
]

for idx, question in enumerate(questions):
    time.sleep(5)

    answer = conversational_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]

    print(f"\n{idx+1}: {question}\n\n{answer}\n")

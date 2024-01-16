from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import json

model_name = "/root/huggingface/models/BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda:3"}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# embeddings = embeddings_model.embed_documents(
#     [
#         "Hi there!",
#         "Oh, hello!",
#         "What's your name?",
#         "My friends call me World",
#         "Hello World!"
#     ]
# )
# print(len(embeddings), len(embeddings[0]))

# embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
# print(embedded_query[:5])

loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# print(docs)

# vector_db = Milvus.from_documents(
#     docs,
#     embeddings_model,
#     connection_args={
#         "host": "192.168.3.226", 
#         "port": "30041",
#     },
# )

# vector_db = Milvus(
#     embedding_function = embeddings_model,
#     connection_args= {
#         "host": "192.168.3.226",
#         "port": "30041",
#         # "user": "minioadmin",
#         # "password": "minioadmin",
#     },
# )

vector_db = FAISS.from_documents(docs, embeddings_model)
# vector_db.add_documents(docs)

query = "What did the president say about Officer Mora"
docs = vector_db.similarity_search(query)

print("================================")
print(docs[0].page_content)

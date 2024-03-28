from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
import json

model_name = "/root/huggingface/models/BAAI/bge-large-en-v1.5"
embedding_device = "cuda:1"
model_kwargs = {"device": embedding_device}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = TextLoader("langchain/rag/state_of_the_union.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=400, add_start_index=False
)
docs = text_splitter.split_documents(documents)

vector_db = FAISS.from_documents(docs, embeddings_model)

query = "what happend to Joshua Davis"
# docs = vector_db.similarity_search(query, k=1)
embedding_vectors = embeddings_model.embed_query(query)
docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)

print("-----------------------------------------------------------------")
print(docs[0].page_content)

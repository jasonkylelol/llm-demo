from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS

model_name = "/root/huggingface/models/BAAI/bge-large-en-v1.5"
embedding_device = "cuda:1"
model_kwargs = {"device": embedding_device}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_splitter = SemanticChunker(embeddings_model,
    breakpoint_threshold_type="percentile",
    # breakpoint_threshold_type="standard_deviation",
    # breakpoint_threshold_type="interquartile",
)

with open("langchain/rag/opanai_lawsuit.txt") as f:
    content = f.read()

docs = text_splitter.create_documents([content])
# print(docs[0].page_content)
print(len(docs))

query = "Why Musk is indispensable to the development of the OpenAI?"
embedding_vectors = embeddings_model.embed_query(query)
vector_db = FAISS.from_documents(docs, embeddings_model)
docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)
print(docs[0].page_content)

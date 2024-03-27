# langchain implementation
## serve for HuggingFaceH4/zephyr-7b-beta model, backed with huggingface text-generation-inference api
- hf_tgi_demo.py contains chat model(Llama2Chat) and default chain(LLMChain) implementation
- custom_chain.py contains custom chain with llm(HuggingFaceTextGenInference) implementation, and custom ChatPromptTemplate and CallBack
- rag.py contains RAG implementation, use embeddings model and vector store for context retrieval
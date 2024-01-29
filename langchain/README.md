# langchain implementation
## serve for HuggingFaceH4/zephyr-7b-beta model, backed with huggingface text-generation-inference api
- langchain_hf_tgi_demo.py contains chat model(Llama2Chat) and default chain(LLMChain) implementation
- langchain_custom_demo.py contains custom chain with llm(HuggingFaceTextGenInference) implementation, and custom ChatPromptTemplate and CallBack
- langchain_retriever.py contains RAG implementation, use embeddings model and vector store for context retrieval
- langchain_agent.py use local model without function calling to implement llm chat agent
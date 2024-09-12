from harbor.apulis.cn:8443/cicd/ops/python:3.10.11

run echo "export PATH=\"/root/.local/bin:$PATH\"" >> /root/.bashrc
run echo "alias ls='ls --color=auto'" >> /root/.bashrc
run echo "alias ll='ls -alF'" >> /root/.bashrc
run echo "alias la='ls -A'" >> /root/.bashrc
run echo "alias l='ls -CF'" >> /root/.bashrc

run sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
run apt update && apt install -y vim curl wget

workdir /workspace

run pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple langchain-text-splitters==0.2.2
add requirements.txt requirements.txt
run pip install -r requirements.txt
run pip uninstall -y graphrag

add graphrag graphrag
add template template
add template_zh template_zh

# graphrag/graphrag/index/graph/extractors/graph/prompts.py
# graphrag/graphrag/index/verbs/text/chunk/strategies/tokens.py
# graphrag/graphrag/index/verbs/text/chunk/strategies/chinese_recursive_text_splitter.py
# graphrag/graphrag/index/graph/extractors/graph/graph_extractor.py
# graphrag/graphrag/index/verbs/entities/extraction/strategies/graph_intelligence/run_graph_intelligence.py
# graphrag/graphrag/config/models/entity_extraction_config.py
# graphrag/graphrag/config/create_graphrag_config.py

# graphrag/graphrag/query/structured_search/local_search/search.py
# graphrag/graphrag/config/models/local_search_config.py
# graphrag/graphrag/query/factories.py
# graphrag/graphrag/config/create_graphrag_config.py

# graphrag/graphrag/query/structured_search/global_search/search.py
# graphrag/graphrag/config/models/global_search_config.py
# graphrag/graphrag/query/factories.py
# graphrag/graphrag/config/create_graphrag_config.py

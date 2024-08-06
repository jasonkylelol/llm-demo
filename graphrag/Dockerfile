from harbor.apulis.cn:8443/cicd/ops/python:3.10.11

add install_poetry.py install_poetry.py
run python install_poetry.py

run echo "export PATH=\"/root/.local/bin:$PATH\"" >> /root/.bashrc
run echo "alias ls='ls --color=auto'" >> /root/.bashrc
run echo "alias ll='ls -alF'" >> /root/.bashrc
run echo "alias la='ls -A'" >> /root/.bashrc
run echo "alias l='ls -CF'" >> /root/.bashrc

workdir /workspace

add graphrag graphrag
run cd graphrag && /root/.local/bin/poetry install

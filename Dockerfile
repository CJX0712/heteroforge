FROM python:3.13-slim

LABEL maintainer="晨星"
LABEL org.opencontainers.image.description="HeteroForge: homophily-aware adaptive routing for graph representation learning"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY heteroforge ./heteroforge
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests
COPY README.md LICENSE pyproject.toml ./

# 默认跑快速自检(字符门禁 + 单测 + 小图冒烟)
CMD ["python", "scripts/verify.py", "--fast"]

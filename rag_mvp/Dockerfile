# RAG MVP — GPU 백엔드 (RTX 5080 / Blackwell → CUDA 12.8 + torch nightly cu128)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch: Blackwell(sm_120) 지원 nightly cu128
RUN pip3 install --break-system-packages --no-cache-dir --pre torch \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# 나머지 의존성
COPY rag_mvp/requirements-rag.txt /app/rag_mvp/requirements-rag.txt
RUN pip3 install --break-system-packages --no-cache-dir -r rag_mvp/requirements-rag.txt \
    && python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true

# 코드 (체크포인트·벡터DB·모델캐시는 볼륨 마운트 — 이미지에 안 넣음)
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
COPY rag_mvp /app/rag_mvp

EXPOSE 8000
# 컨테이너는 0.0.0.0 바인드 (외부 접근)
CMD ["uvicorn", "rag_mvp.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

# 배포 — Docker (GPU, 켰다/껐다)

로컬 GPU(RTX 5080)에서 컨테이너로 돌리고, 원할 때 켜고 끄는 방식.
GPU-in-Docker는 이 PC에서 **동작 확인됨**(컨테이너에서 RTX 5080 인식).

## 준비 (한 번만)

- Docker Desktop(WSL2) + NVIDIA 드라이버 — 이미 됨
- **호스트에 이미 있어야 하는 것**(볼륨으로 마운트, 이미지에 안 넣음):
  - 색인된 벡터DB: `%USERPROFILE%\.rag_mvp\chroma` (이미 색인 완료 — arXiv/KP20k/mixed)
  - 모델 캐시: `%USERPROFILE%\.cache\huggingface` (bge, Qwen — 이미 다운로드됨)
  - 체크포인트: `outputs\checkpoints\{keybart_full, reranker_scibert}` (신규 문서 색인 시만)

## 이미지 빌드 (최초 1회, ~20~40분)

```powershell
cd rag_mvp
docker compose build
```
> torch(cu128 nightly) + 의존성 설치로 시간이 걸리고 이미지가 큽니다(~10GB). 한 번 빌드하면 이후엔 캐시됨.

## 켜기 / 끄기 (원할 때)

```powershell
cd rag_mvp

docker compose up -d      # 켜기  → http://localhost:8000
docker compose stop       # 끄기  (컨테이너 정지, 상태 유지)
docker compose start      # 다시 켜기 (빠름)

docker compose logs -f    # 로그 보기 (모델 로딩·요청)
docker compose down       # 완전 제거 (벡터DB·캐시는 볼륨이라 보존됨)
```

- **켜기**: `up -d` 또는 이후엔 `start` → 브라우저에서 `http://localhost:8000`
- **끄기**: `stop` (다음에 `start`로 즉시 재개) 또는 `down`(컨테이너 삭제, 데이터는 남음)
- GPU·포트는 컨테이너가 떠 있는 동안만 점유 → 끄면 반환

## 접속

- 본인 PC: `http://localhost:8000`
- 같은 네트워크 팀원: `http://<내PC_IP>:8000` (compose가 0.0.0.0:8000 바인드)
- 인터넷 공개(선택): `cloudflared tunnel --url http://localhost:8000` → 공개 URL

## 참고 / 주의

- **프론트는 CDN(React/Tailwind)** 사용 → 보는 브라우저에 인터넷 필요.
- **모델·벡터DB는 마운트**라 이미지에 없음 → 컨테이너는 가볍게 뜨고, 데이터는 호스트에 영속.
- 첫 요청 시 모델 GPU 로드로 몇 초 지연(워밍업).
- 신규 문서 색인(`/api/ingest`)은 체크포인트 마운트가 있어야 동작.
- 프로덕션 공개 시: 인증 추가·CORS 축소 권장 (지금은 데모 기준 열림).

## 빠른 점검

```powershell
docker compose up -d
curl http://localhost:8000/api/health   # {"corpora":{arxiv,kp20k,mixed}, ...}
```

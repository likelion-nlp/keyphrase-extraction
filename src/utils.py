"""공용 유틸리티: 시드 고정, 디바이스 선택, 파일 I/O, 실험 로깅.

모든 노트북과 스크립트는 이 모듈의 함수를 통해 환경을 초기화한다.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def set_seed(seed: int = 42) -> None:
    """Python/NumPy/PyTorch 시드를 모두 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(verbose: bool = True):
    """CUDA GPU가 있으면 GPU, 없으면 CPU를 반환한다.

    이 프로젝트의 모든 임베딩·학습·생성은 GPU 사용을 전제로 설계되었다.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[device] GPU: {name} ({vram:.1f} GB)")
            print(f"[device] bf16 지원: {torch.cuda.is_bf16_supported()}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device] GPU를 찾지 못했습니다. CPU로 실행합니다 (학습/생성은 매우 느립니다).")
    return device


def print_env_info() -> dict:
    """주요 라이브러리 버전과 GPU 정보를 출력하고 dict로 반환한다."""
    import importlib.metadata as md

    import torch

    info: dict[str, Any] = {"python": os.sys.version.split()[0]}
    for pkg in [
        "torch", "transformers", "datasets", "sentence-transformers",
        "keybert", "scikit-learn", "nltk", "numpy", "pandas",
    ]:
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = "not installed"
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    for k, v in info.items():
        print(f"{k:>24}: {v}")
    return info


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any):
    """numpy 스칼라/배열을 네이티브 타입으로 직렬화한다.

    default=str만 쓰면 np.float32 점수가 문자열("0.12")로 저장되어
    다시 읽을 때 연산이 깨진다 — 반드시 숫자는 숫자로 저장한다.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config(path: str | Path) -> dict:
    """configs/*.yaml 파일을 dict로 로드한다."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ExperimentLogger:
    """실험 결과를 CSV 한 줄로 누적 기록한다 (마스터 플랜 14절).

    사용 예::

        logger = ExperimentLogger()
        logger.log(run_id="keybart_beam5_seed42", model="bloomberg/KeyBART",
                   seed=42, f1_at_5=0.31)
        logger.to_dataframe()
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else OUTPUTS_DIR / "metrics" / "experiments.csv"
        ensure_dir(self.path.parent)

    def log(self, **row: Any) -> dict:
        row.setdefault("logged_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        rows = self._read_all()
        # 같은 run_id는 최신 결과로 덮어쓴다
        run_id = row.get("run_id")
        if run_id is not None:
            rows = [r for r in rows if r.get("run_id") != run_id]
        rows.append({k: v for k, v in row.items()})
        fieldnames: list[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return row

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def to_dataframe(self):
        import pandas as pd

        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)


class Timer:
    """with 블록 실행 시간을 측정한다."""

    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"[timer] {self.name}: {self.elapsed:.2f}s")

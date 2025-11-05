"""
Streamlit App: 手の骨格（MediaPipe Hands）で2人の同期度をDTW×コサインで算出

使い方:
  $ pip install -U streamlit opencv-python mediapipe==0.10.14 numpy pandas
  $ streamlit run sync_hands_app.py

できること:
- 2つの動画（A/B）をアップロードして、手のランドマーク時系列を抽出
- 部位ベクトル（例: 手首→人差し指MCP など）を作り、コサイン類似度を局所コストとしてDTW
- パス長で平均化したDTW距離 d̄∈[0,1] から同期度 Sync% = 100×(1−d̄) を算出
- 欠損は NaN埋め→マスクして計算（単純）
- 処理を軽くするためのフレーム間引き（frame_stride）

注意:
- このアプリは"解析同期度"用の最小構成です（描画・動画出力は別アプリに任せる想定）
- MediaPipe Handsは各フレームの瞬間値のみ返すため、時系列化は実装側で行っています
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# MediaPipe 0.10.x
import mediapipe as mp

# -----------------------------
# 定数 / ユーティリティ
# -----------------------------
HAND_JOINTS = {
     0:  "手首",
    1:  "親指 CMC",  2:  "親指 MCP",  3:  "親指 IP",   4:  "親指 先端",
    5:  "人差し指 MCP", 6:  "人差し指 PIP", 7:  "人差し指 DIP", 8:  "人差し指 先端",
    9:  "中指 MCP",     10: "中指 PIP",     11: "中指 DIP",     12: "中指 先端",
    13: "薬指 MCP",     14: "薬指 PIP",     15: "薬指 DIP",     16: "薬指 先端",
    17: "小指 MCP",     18: "小指 PIP",     19: "小指 DIP",     20: "小指 先端",
}

DEFAULT_SEGMENTS = [
    (0, 5),   # WRIST -> INDEX_MCP
    (0, 9),   # WRIST -> MIDDLE_MCP
    (5, 8),   # INDEX_MCP -> INDEX_TIP
    (9, 12),  # MIDDLE_MCP -> MIDDLE_TIP
]

@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


def get_video_meta(cap: cv2.VideoCapture) -> VideoMeta:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-6:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMeta(w, h, fps, n)

# -----------------------------
# 抽出: Hands の時系列
# -----------------------------

def extract_timeseries_hands(results, image_shape=None, pixel=False) -> Dict[str, np.ndarray]:
    """MediaPipe Hands の1フレーム出力から左右21点×(x,y,z) を返す。
    返り値: {"left": (21,3) float32, "right": (21,3) float32} （検出なしはNaN）
    pixel=True & image_shape=(H,W,...) のとき x,y をピクセル座標化（zはそのまま）
    """
    out = {
        "left":  np.full((21, 3), np.nan, dtype=np.float32),
        "right": np.full((21, 3), np.nan, dtype=np.float32),
    }
    if not results or not getattr(results, "multi_hand_landmarks", None):
        return out

    handed_list = getattr(results, "multi_handedness", None)
    if handed_list is None:
        # ハンドネス情報が無い場合の保険
        iter_src = zip(results.multi_hand_landmarks, ["Left", "Right"])
    else:
        iter_src = zip(results.multi_hand_landmarks, handed_list)

    for lms, handed in iter_src:
        label = handed if isinstance(handed, str) else handed.classification[0].label  # "Left"/"Right"
        label = label.lower()
        arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
        if pixel and image_shape is not None:
            H, W = image_shape[:2]
            arr[:, 0] *= W
            arr[:, 1] *= H
        out[label] = arr
    return out


def build_hands_timeseries(video_path: str, frame_stride: int = 1) -> Dict[str, np.ndarray]:
    """動画からHandsの時系列を生成。
    戻り値:
      {
        "t": (T,),
        "left": (T,21,3),
        "right": (T,21,3),
        "meta": VideoMeta,
      }
    検出できないフレームはNaNで埋める。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: " + video_path)
    meta = get_video_meta(cap)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    t_list, L_list, R_list = [], [], []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            out = extract_timeseries_hands(res, image_shape=frame.shape, pixel=False)
            t_list.append(idx / meta.fps)
            L_list.append(out["left"])  # (21,3)
            R_list.append(out["right"]) # (21,3)
            idx += 1
    finally:
        cap.release()
        hands.close()

    t = np.array(t_list, dtype=np.float32)
    L = np.stack(L_list, axis=0) if len(L_list)>0 else np.zeros((0,21,3), np.float32)
    R = np.stack(R_list, axis=0) if len(R_list)>0 else np.zeros((0,21,3), np.float32)
    return {"t": t, "left": L, "right": R, "meta": meta}

# -----------------------------
# ベクトル化 / 正規化
# -----------------------------

def make_segment_vectors(seq: np.ndarray, seg: Tuple[int,int]) -> np.ndarray:
    """(T,21,3) から部位ベクトル (to-from) を作る → (T,3)
    欠損(NaN)はそのまま（後でマスク）
    """
    a, b = seg
    v = seq[:, b, :] - seq[:, a, :]
    return v


def unit_vectors(v: np.ndarray) -> np.ndarray:
    """(T,3) を単位ベクトル化（ゼロ割回避）"""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + 1e-8)

# -----------------------------
# DTW（シンプル版 + 帯制約）
# -----------------------------

def dtw_avg_cost(X: np.ndarray, Y: np.ndarray, band: Optional[int] = None) -> float:
    """局所コスト in [0,1] を前提とした DTW の平均コスト d̄ を返す。
    X,Y: 形 (Tx, F), (Ty, F) の特徴（ここでは F=3 の単位ベクトル等）
    局所コスト: (1 - cos)/2 を使用
    band: Sakoe-Chiba 帯幅（時間ステップ単位）。Noneなら無制限
    """
    Tx, Ty = len(X), len(Y)
    # cos類似度 → コスト行列をオンザフライで計算する代わりに逐次で
    INF = 1e9
    D = np.full((Tx + 1, Ty + 1), INF, dtype=np.float32)
    D[0, 0] = 0.0

    for i in range(1, Tx + 1):
        # 帯制約
        j_start = 1
        j_end = Ty + 1
        if band is not None:
            j_start = max(1, i - band)
            j_end = min(Ty + 1, i + band + 1)
        for j in range(j_start, j_end):
            # 局所コスト c = (1 - cos)/2 ∈ [0,1]
            xi = X[i - 1]
            yj = Y[j - 1]
            cos = float(np.dot(xi, yj))  # 単位ベクトル前提
            c = 0.5 * (1.0 - max(min(cos, 1.0), -1.0))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # パス長を概算: 最短でも max(Tx,Ty)、最長で Tx+Ty
    # ここでは動的復元を省略し、近似として (Tx+Ty)/2 を用いることもできるが、
    # より妥当な平均化のため minコスト方向から逆追跡でLを数える
    i, j = Tx, Ty
    L = 0
    while i > 0 or j > 0:
        L += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # 3近傍から逆遷移
            idx = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if idx == 0:
                i -= 1
            elif idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    avg = float(D[Tx, Ty] / max(L, 1))
    return avg

# -----------------------------
# 同期度算出
# -----------------------------

def sync_score_vectors(VA: np.ndarray, VB: np.ndarray, band: Optional[int] = None) -> float:
    """部位ベクトル時系列 VA, VB (T,3) から同期度%を返す。
    1) 単位化 2) 欠損マスク 3) DTW平均コスト→ Sync = 100×(1−d̄)
    """
    # 欠損があるフレームを共通にマスクアウト
    mask = ~np.any(np.isnan(VA) | np.isnan(VB), axis=1)
    VA2 = VA[mask]
    VB2 = VB[mask]
    if len(VA2) < 2 or len(VB2) < 2:
        return float('nan')
    UA = unit_vectors(VA2)
    UB = unit_vectors(VB2)
    d_bar = dtw_avg_cost(UA, UB, band=band)  # [0,1]
    return max(0.0, 100.0 * (1.0 - d_bar))


def overall_sync_from_segments(A: np.ndarray, B: np.ndarray, segments: List[Tuple[int,int]], band: Optional[int]=None) -> Tuple[pd.DataFrame, float]:
    """左右どちらかの手の (T,21,3) ×2（A/B）と、セグメント一覧から
    各セグメントの同期度テーブルと等重み平均の総合スコアを返す。
    """
    rows = []
    scores = []
    for (a, b) in segments:
        name = f"{HAND_JOINTS[a]}→{HAND_JOINTS[b]}"
        va = make_segment_vectors(A, (a, b))
        vb = make_segment_vectors(B, (a, b))
        s = sync_score_vectors(va, vb, band=band)
        rows.append({"segment": name, "score(%)": s})
        if not np.isnan(s):
            scores.append(s)
    df = pd.DataFrame(rows)
    overall = float(np.mean(scores)) if len(scores) else float('nan')
    return df, overall

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Hands同期度 (DTW×Cos)", layout="wide")
st.title("手の同期度解析（MediaPipe Hands × DTW × コサイン）")
st.caption("2動画の手の動きの一致度をパーセンテージで算出します。")

with st.sidebar:
    st.header("設定")
    frame_stride = st.select_slider("フレーム間引き", options=[1,2,3,4,5], value=1)
    band = st.select_slider("Sakoe–Chiba帯幅（制約）", options=[None, 5, 10, 15, 20, 30], value=10)
    hand_side = st.selectbox("どの手で比較するか", ["left", "right"], index=0)
    st.markdown("**セグメント（部位）**を選んでください：")
    # セグメント選択（簡易）
    chosen = st.multiselect(
        "ベクトル（from→to）",
        options=[f"{HAND_JOINTS[a]}→{HAND_JOINTS[b]}" for (a,b) in DEFAULT_SEGMENTS],
        default=[f"{HAND_JOINTS[a]}→{HAND_JOINTS[b]}" for (a,b) in DEFAULT_SEGMENTS],
    )
    segments = []
    for s in chosen:
        for (a,b) in DEFAULT_SEGMENTS:
            if s == f"{HAND_JOINTS[a]}→{HAND_JOINTS[b]}":
                segments.append((a,b))
                break

colA, colB = st.columns(2)
with colA:
    st.subheader("動画A")
    upA = st.file_uploader("Aをアップロード (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"], key="A")
    if upA is not None:
        st.video(upA)
with colB:
    st.subheader("動画B")
    upB = st.file_uploader("Bをアップロード (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"], key="B")
    if upB is not None:
        st.video(upB)

run = st.button("同期度を計算", type="primary", disabled=(upA is None or upB is None or len(segments)==0))

if run and upA is not None and upB is not None and len(segments)>0:
    try:
        with tempfile.TemporaryDirectory() as td:
            pA = os.path.join(td, "A")
            pB = os.path.join(td, "B")
            # previewでポインタが進んでいる可能性 → リセット
            try:
                upA.seek(0); upB.seek(0)
            except Exception:
                pass
            with open(pA, "wb") as f: f.write(upA.getbuffer())
            with open(pB, "wb") as f: f.write(upB.getbuffer())

            st.info("MediaPipe Handsで時系列を抽出中…")
            seqA = build_hands_timeseries(pA, frame_stride=frame_stride)
            seqB = build_hands_timeseries(pB, frame_stride=frame_stride)

            A = seqA[hand_side]  # (T,21,3)
            B = seqB[hand_side]

            st.info("同期度を計算中…")
            df, overall = overall_sync_from_segments(A, B, segments, band=band if isinstance(band,int) else None)

            st.success("計算完了")
            st.subheader("部位別スコア")
            st.dataframe(df, use_container_width=True)
            st.markdown(f"**総合同期度（等重み）: {overall:.1f}%**")

    except Exception as e:
        st.error(f"エラー: {e}\n\n対処案:\n- フレーム間引きを増やす（処理軽量化）\n- 短い/小さい動画で試す（事前にリサイズ）\n- MediaPipe/OpenCVの更新: pip install -U mediapipe opencv-python")
else:
    st.info("左右の手・部位を選んで2本の動画をアップし、『同期度を計算』を押してください。")

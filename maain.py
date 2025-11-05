"""
Hands Only: MediaPipe Hands ▶ MP4/AVI 書き出し + 同期度(DTW×DBA)

Usage:
  $ pip install -U streamlit opencv-python mediapipe==0.10.14 numpy pandas tslearn==0.6.3 scipy
  $ streamlit run streamlit_hands_sync_app.py

Notes:
- MP4が失敗する環境ではサイドバーで AVI に切り替えてください（自動フォールバックも有り）。
- CSVは書き出しません（動画DLと表/グラフ表示のみ）。
"""

from __future__ import annotations
import os
import time
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# MediaPipe Hands (0.10.x)
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# DTW/DBA
from tslearn.metrics import dtw as ts_dtw
from tslearn.barycenters import dtw_barycenter_averaging as ts_dba

# -----------------------------
# Helpers
# -----------------------------
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
    if fps <= 1e-6: fps = 30.0
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMeta(w, h, fps, f)

def aspect_resize(w: int, h: int, target_w: Optional[int]) -> Tuple[int,int]:
    if not target_w or target_w <= 0: return w, h
    s = target_w / float(w)
    return target_w, int(round(h * s))

def make_writer(path: str, size: Tuple[int,int], fps: float, use_mp4: bool) -> cv2.VideoWriter:
    """
    コーデック自動フォールバック:
      MP4: mp4v -> avc1 -> H264 -> (失敗時) AVI(XVID)
      AVI: XVID
    """
    candidates = [("XVID",".avi")] if not use_mp4 else [("mp4v",".mp4"),("avc1",".mp4"),("H264",".mp4")]
    last_err = None
    for fourcc_str, ext in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out_path = path if path.endswith(ext) else path + ext
        w = cv2.VideoWriter(out_path, fourcc, fps, size)
        if w.isOpened(): return w
        last_err = f"FOURCC {fourcc_str} failed for {out_path}"
    # 最終フォールバック AVI
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = path if path.endswith(".avi") else path + ".avi"
    w = cv2.VideoWriter(out_path, fourcc, fps, size)
    if w.isOpened(): return w
    raise RuntimeError(f"Failed to open video writer. {last_err or ''} Try AVI or install codecs.")

# -----------------------------
# 書き出し（Handsのみ）
# -----------------------------
def process_video_hands(
    input_path: str,
    out_base_path: str,
    draw_landmarks: bool = True,
    max_num_hands: int = 2,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
    output_width: Optional[int] = None,
    use_mp4: bool = True,
    update_progress=None,
) -> Tuple[str, VideoMeta]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError("Could not open the uploaded video.")
    meta = get_video_meta(cap)
    out_w, out_h = aspect_resize(meta.width, meta.height, output_width)

    writer = make_writer(out_base_path, (out_w, out_h), meta.fps, use_mp4)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    idx = 0; last = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if draw_landmarks and res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        bgr, lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(bgr)
            idx += 1
            if update_progress and (time.time()-last)>0.1 and meta.frame_count>0:
                update_progress(min(idx/meta.frame_count, 1.0))
                last = time.time()
    finally:
        cap.release(); writer.release(); hands.close()

    # どちらが書かれたか判定
    mp4, avi = out_base_path + ".mp4", out_base_path + ".avi"
    final = mp4 if os.path.exists(mp4) else (avi if os.path.exists(avi) else out_base_path + (".mp4" if use_mp4 else ".avi"))
    return final, meta

# -----------------------------
# Hands → 時系列(126次元: 左手63 + 右手63)
# -----------------------------
def _flatten_hands_lr_from_results(res) -> np.ndarray:
    """
    両手を固定順（Left→Right）・固定次元(21*3*2=126)の1Dベクトルに。
    無い手はゼロ埋め。world_landmarks優先、無ければimage座標(x,y,z)。
    """
    L = np.zeros((21, 3), dtype=np.float32)
    R = np.zeros((21, 3), dtype=np.float32)

    # 取得（NoneならNoneのまま）
    wls = getattr(res, "multi_hand_world_landmarks", None)   # list[LandmarkList] or None
    ils = getattr(res, "multi_hand_landmarks", None)         # list[NormalizedLandmarkList] or None
    hds = getattr(res, "multi_handedness", None)             # list[ClassificationList] or None

    hands_list = []

    def _append_from_lists(lm_lists, handedness_lists):
        if not lm_lists:
            return
        n = len(lm_lists) if not handedness_lists else min(len(lm_lists), len(handedness_lists))
        for i in range(n):
            lm_list = lm_lists[i]
            pts = getattr(lm_list, "landmark", None)  # ★ LandmarkListは .landmark が配列
            if pts is None:
                continue
            # handedness が無い/ズレている場合のフォールバック
            label = "Left" if i == 0 else "Right"
            if handedness_lists and i < len(handedness_lists):
                try:
                    label = handedness_lists[i].classification[0].label  # "Left"/"Right"
                except Exception:
                    pass
            coords = [(p.x, p.y, p.z) for p in pts]
            hands_list.append((label, coords))

    # 優先: world → 次: image
    _append_from_lists(wls, hds)
    if not hands_list:
        _append_from_lists(ils, hds)

    def fix21x3(arr_like):
        arr = np.array(arr_like, dtype=np.float32)
        if arr.shape != (21, 3):
            out = np.zeros((21, 3), dtype=np.float32)
            out[:min(21, arr.shape[0]), :min(3, arr.shape[1])] = arr[:min(21, arr.shape[0]), :min(3, arr.shape[1])]
            return out
        return arr

    # Left→Right に固定（無い側はゼロ）
    for label, coords in hands_list:
        if label.lower().startswith("left"):
            L = fix21x3(coords)
        elif label.lower().startswith("right"):
            R = fix21x3(coords)

    return np.concatenate([L.reshape(-1), R.reshape(-1)], axis=0)  # (126,)

# -----------------------------
# DBA → DTW → SyncScore
# -----------------------------
def compute_sync_scores(seqs: list[np.ndarray]) -> dict:
    """
    seqs: list of (T_i, 126)
    return: {"barycenter": (Tb,126), "dtw_dist": [...], "scores": [...], "max_dist": float}
    """
    norm = []
    for X in seqs:
        if X.ndim!=2 or X.shape[0]==0:
            norm.append(np.zeros((1,1), dtype=np.float32)); continue
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        norm.append(((X - mu) / sd).astype(np.float32))

    B = ts_dba(norm)  # (Tb,126)
    dists = [float(ts_dtw(X, B)) for X in norm]
    maxd = max(dists) if dists else 1.0
    if maxd < 1e-9: maxd = 1e-9
    scores = [(1.0 - d/maxd) * 100.0 for d in dists]
    return {"barycenter": B, "dtw_dist": dists, "scores": scores, "max_dist": maxd}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hands限定：骨格検出＆同期度", layout="wide")
st.title("Hands限定 骨格ダウンロード ")
st.caption("検出は手だけ（MediaPipe Hands）。CSVは出力しません。")

with st.sidebar:
    st.header("書き出し設定（Hands）")
    draw_landmarks = st.checkbox("ランドマークを描画", value=True)
    output_width = st.number_input("出力幅(px)（空なら元サイズ）", min_value=0, max_value=4096, value=0, step=32)
    max_num_hands = st.slider("最大手数", 1, 2, 2)
    det_conf = st.slider("検出信頼度", 0.1, 0.9, 0.5, 0.05)
    track_conf = st.slider("追跡信頼度", 0.1, 0.9, 0.5, 0.05)
    container_fmt = st.selectbox("書き出しコンテナ", ["MP4 (mp4v)", "AVI (XVID)"])

uploaded = st.file_uploader("動画ファイルをアップロード (mp4/avi/mov/mkv)", type=["mp4","mov","avi","mkv"])

col1, col2 = st.columns(2)
with col1:
    if uploaded is not None:
        st.video(uploaded, autoplay=False)

run = st.button("この設定で処理する", type="primary", disabled=(uploaded is None))

if run and uploaded is not None:
    use_mp4 = container_fmt.startswith("MP4")
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input")
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            base_out = os.path.join(td, "output")
            prog = st.progress(0.0, text="処理中…")
            def _update(p): prog.progress(p, text=f"処理中… {int(p*100)}%")

            with st.spinner("MediaPipe Handsで推論中…"):
                out_path, meta = process_video_hands(
                    input_path=in_path,
                    out_base_path=base_out,
                    draw_landmarks=draw_landmarks,
                    max_num_hands=max_num_hands,
                    det_conf=det_conf,
                    track_conf=track_conf,
                    output_width=(output_width or None),
                    use_mp4=use_mp4,
                    update_progress=_update,
                )
            prog.empty()

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.success(f"完了：{os.path.basename(out_path)}  |  元: {meta.width}x{meta.height}@{meta.fps:.1f}fps  |  フレーム数: {meta.frame_count}")
            col_v, col_d = st.columns([2,1])
            with col_v:
                st.subheader("プレビュー")
                st.video(video_bytes)
            with col_d:
                st.subheader("ダウンロード")
                st.download_button(
                    label="動画をダウンロード",
                    data=video_bytes,
                    file_name=os.path.basename(out_path),
                    mime="video/mp4" if use_mp4 else "video/avi",
                )
    except Exception as e:
        st.error(
            f"エラー: {e}\n\n対処案:\n"
            "- '書き出しコンテナ' を 'AVI' に変更\n"
            "- 別の動画で試す / 出力幅を下げる\n"
            "- OpenCV/MediaPipe を更新: pip install -U opencv-python mediapipe\n"
            "- Windowsは“HEVC Video Extensions”導入も検討"
        )
else:
    st.info("動画をアップロードしてから『この設定で処理する』を押してください。")


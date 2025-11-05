"""
Streamlit: Skeleton (Pose/Hands/Holistic) Detection on Video and MP4 Export

Usage:
  $ pip install -U streamlit opencv-python mediapipe==0.10.14 numpy
  $ streamlit run streamlit_skeleton_detection_app.py

Notes:
- If MP4 writing fails on your environment, set the sidebar option to write AVI instead.
- On Windows, OpenCV sometimes needs additional codecs. As a fallback, choose AVI.
- This app DOES NOT write CSV (video only), as requested.
"""

from __future__ import annotations
import os
import io
import time
import math
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Mediapipe imports (0.10.x)
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# -----------------------------
# Small helpers
# -----------------------------
@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


def get_video_meta(cap: cv2.VideoCapture) -> VideoMeta:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-6:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMeta(width, height, fps, frame_count)


def aspect_resize(w: int, h: int, target_w: Optional[int]) -> Tuple[int, int]:
    if not target_w or target_w <= 0:
        return w, h
    scale = target_w / float(w)
    return target_w, int(round(h * scale))


def make_writer(path: str, size: Tuple[int, int], fps: float, use_mp4: bool) -> cv2.VideoWriter:
    if use_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ext = ".mp4"
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        ext = ".avi"
    if not path.endswith(ext):
        path += ext
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer. Try switching the container (MP4/AVI) in the sidebar.")
    return writer


# -----------------------------
# Core processing
# -----------------------------

def process_video(
    input_path: str,
    out_base_path: str,
    detector: str = "Hands",
    draw_landmarks: bool = True,
    max_num_hands: int = 2,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
    model_complexity: int = 1,
    output_width: Optional[int] = None,
    use_mp4: bool = True,
    update_progress=None,
) -> Tuple[str, VideoMeta]:
    """Process a video with MediaPipe and return output file path + meta."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    meta = get_video_meta(cap)
    out_w, out_h = aspect_resize(meta.width, meta.height, output_width)

    # Prepare writer
    out_path = out_base_path
    writer = make_writer(out_path, (out_w, out_h), meta.fps, use_mp4)

    # Create solution by detector type
    if detector == "Hands":
        solution = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
    elif detector == "Pose":
        solution = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
    elif detector == "Holistic":
        solution = mp.solutions.holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
    else:
        cap.release()
        writer.release()
        raise ValueError(f"Unknown detector: {detector}")

    frame_idx = 0
    last_update = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Process frame
            results = solution.process(frame_rgb)

            # Draw landmarks if requested
            if draw_landmarks:
                if detector == "Hands" and results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            hand_lms,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                elif detector == "Pose" and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )
                elif detector == "Holistic":
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.face_landmarks,
                            mp.solutions.holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                        )
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.left_hand_landmarks,
                            mp.solutions.holistic.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.right_hand_landmarks,
                            mp.solutions.holistic.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.pose_landmarks,
                            mp.solutions.holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                        )

            # Resize if needed
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (out_w, out_h):
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(frame_bgr)
            frame_idx += 1

            # Progress update throttled to ~10 Hz
            if update_progress and (time.time() - last_update) > 0.1 and meta.frame_count > 0:
                update_progress(min(frame_idx / meta.frame_count, 1.0))
                last_update = time.time()

    finally:
        cap.release()
        writer.release()
        solution.close()

    ext = ".mp4" if use_mp4 else ".avi"
    final_path = out_base_path + ext
    return final_path, meta


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Skeleton Detection (MediaPipe)", layout="wide")

st.title("骨格検出（MediaPipe）▶︎ Streamlitで動画を書き出し")
st.caption("CSVは出力しません。ランドマーク重畳済みの動画のみを保存・ダウンロードできます。")

with st.sidebar:
    st.header("設定")
    detector = st.selectbox("検出タイプ", ["Hands", "Pose", "Holistic"], index=0, help="手だけ / 全身 / 顔+手+ポーズ")
    draw_landmarks = st.checkbox("ランドマークを描画", value=True)
    output_width = st.number_input("出力幅(px)（空なら元サイズ）", min_value=0, max_value=4096, value=0, step=32)

    st.subheader("詳細設定")
    max_num_hands = st.slider("最大手数 (Hands)", 1, 4, 2)
    det_conf = st.slider("検出信頼度", 0.1, 0.9, 0.5, 0.05)
    track_conf = st.slider("追跡信頼度", 0.1, 0.9, 0.5, 0.05)
    model_complexity = st.select_slider("モデル複雑度 (Pose/Holistic)", options=[0,1,2], value=1)
    container_fmt = st.selectbox("書き出しコンテナ", ["MP4 (mp4v)", "AVI (XVID)"])

uploaded = st.file_uploader("動画ファイルをアップロード (mp4/avi/mov/mkv)", type=["mp4","mov","avi","mkv"]) 

col1, col2 = st.columns(2)

with col1:
    if uploaded is not None:
        st.video(uploaded, autoplay=False)

run = st.button("この設定で処理する", type="primary", disabled=(uploaded is None))

if run and uploaded is not None:
    use_mp4 = (container_fmt.startswith("MP4"))
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input")
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            base_out = os.path.join(td, "output")

            prog = st.progress(0.0, text="処理中…")
            def _update(p):
                prog.progress(p, text=f"処理中… {int(p*100)}%")

            with st.spinner("MediaPipeで推論中…"):
                out_path, meta = process_video(
                    input_path=in_path,
                    out_base_path=base_out,
                    detector=detector,
                    draw_landmarks=draw_landmarks,
                    max_num_hands=max_num_hands,
                    det_conf=det_conf,
                    track_conf=track_conf,
                    model_complexity=model_complexity,
                    output_width=(output_width or None),
                    use_mp4=use_mp4,
                    update_progress=_update,
                )
            prog.empty()

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.success(
                f"完了：{os.path.basename(out_path)}  |  元: {meta.width}x{meta.height}@{meta.fps:.1f}fps  |  フレーム数: {meta.frame_count}")

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
        st.error(f"エラー: {e}\n\n対処案:\n- サイドバーで '書き出しコンテナ' を 'AVI' に変更\n- 別の動画で試す / 解像度を下げる (出力幅)\n- OpenCV/MediaPipe を最新化: pip install -U opencv-python mediapipe\n- GPUではなくCPUでの実行を想定しています")
else:
    st.info("左の設定を調整し、動画をアップロードしてから『この設定で処理する』を押してください。")

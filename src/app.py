import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
from collections import defaultdict

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="WasteVision Analytics",
    page_icon="🗑️",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    return YOLO(model_path)

model = load_model()

# ── Header ───────────────────────────────────────────────
st.title("🗑️ WasteVision Analytics")
st.markdown("Road garbage detection — category wise analysis")
st.divider()

# ── Upload section ───────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload")
    file_type = st.radio("Select type", ["Image", "Video"])
    
    if file_type == "Image":
        uploaded = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
    else:
        uploaded = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

# ── Process Image ─────────────────────────────────────────
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    results = model(img_array, conf=0.25)
    
    category_counts = defaultdict(int)
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        category_counts[class_name] += 1
    
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, dict(category_counts)

# ── Process Video ─────────────────────────────────────────
def process_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    tracked_ids = defaultdict(set)
    frame_count = 0
    stframe = st.empty()
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Har 3rd frame process karo — speed ke liye
        if frame_count % 3 == 0:
            results = model.track(frame, persist=True, conf=0.25)
            
            if results[0].boxes.id is not None:
                for box in results[0].boxes:
                    if box.id is not None:
                        track_id = int(box.id)
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        tracked_ids[class_name].add(track_id)
            
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, channels="RGB", use_column_width=True)
        
        progress.progress(min(frame_count / total_frames, 1.0))
    
    cap.release()
    os.unlink(tfile.name)
    
    category_counts = {k: len(v) for k, v in tracked_ids.items()}
    return category_counts

# ── Main Logic ────────────────────────────────────────────
if uploaded is not None:
    
    if file_type == "Image":
        with st.spinner("Analyzing image..."):
            annotated_img, counts = process_image(uploaded)
        
        with col2:
            st.subheader("🔍 Detection Result")
            st.image(annotated_img, use_column_width=True)
    
    else:
        st.subheader("🎬 Video Analysis")
        with st.spinner("Processing video..."):
            counts = process_video(uploaded)
    
    # ── Dashboard ─────────────────────────────────────────
    if counts:
        st.divider()
        st.subheader("📊 Analysis Dashboard")
        
        total = sum(counts.values())
        
        # Metrics row
        metric_cols = st.columns(len(counts) + 1)
        metric_cols[0].metric("🗑️ Total Waste", total)
        
        for i, (cat, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
            metric_cols[i+1].metric(f"{cat.capitalize()}", count)
        
        st.divider()
        
        # Charts
        df = pd.DataFrame({
            'Category': list(counts.keys()),
            'Count': list(counts.values())
        }).sort_values('Count', ascending=False)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_bar = px.bar(
                df, x='Category', y='Count',
                color='Category',
                title="Category wise Garbage Count",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with chart_col2:
            fig_pie = px.pie(
                df, names='Category', values='Count',
                title="Category Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Table
        st.subheader("📋 Detailed Breakdown")
        df['Percentage'] = (df['Count'] / total * 100).round(1).astype(str) + '%'
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Report (CSV)",
            data=csv,
            file_name="wastevision_report.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("⚠️ No garbage detected in this file.")

else:
    with col2:
        st.info("👈 Upload an image or video to start analysis")
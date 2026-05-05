import streamlit as st
import cv2
import torch
import numpy as np
import os
import subprocess
from PIL import Image
from model.model import VideoEnhancementModel

# Page config
st.set_page_config(layout="wide", page_title="Kineura Frame Predictor")

# Load model
@st.cache_resource
def load_model():
    # Initialize model
    model = VideoEnhancementModel()

    # Get best weights
    checkpoint_path = 'checkpoints/best_weights.pth'

    # Try to load weights
    try:
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(state_dict)
        model.to('cuda')
        model.eval() # Set to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Checkpoint not found at {checkpoint_path}. Check your path!")
        return None

# Video handling
video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if video_file:
    # Save uploaded file locally to read with OpenCV
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture("temp_video.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Top section: Triplet slider
    st.header("Step-by-Step Triplet Validation")
    idx = st.slider("Select Triplet Start Frame", 0, total_frames - 3 - 1, 0)

    # Grab a triplet
    frames_bgr = []
    for i in range(idx, idx + 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames_bgr.append(frame)

    cap.release()

    # Frame formatting and frame prediction

    # Load model
    model = load_model()
    pred = None
    if model is not None and len(frames_bgr) >= 2:
        # Resize/Format for model (448x256)
        f1 = cv2.resize(frames_bgr[0], (448, 256))
        f2 = cv2.resize(frames_bgr[1], (448, 256))

        # Color and Tensor conversion
        t1 = torch.from_numpy(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        # Pass to model
        input_tensor = torch.cat([t1, t2], dim=0).unsqueeze(0).to('cuda')
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process back to image
        pred = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

        # Resize back to match original video for display
        h_orig, w_orig = frames_bgr[0].shape[:2]
        pred = cv2.resize(pred, (w_orig, h_orig))

    # Display triplet
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB), caption="Frame 1 (Input)")
    with col2:
        st.image(cv2.cvtColor(frames_bgr[1], cv2.COLOR_BGR2RGB), caption="Frame 2 (Input)")
    with col3:
        st.image(cv2.cvtColor(frames_bgr[2], cv2.COLOR_BGR2RGB), caption="Frame 3 (Original)")
        if pred is not None:
            st.image(pred, caption="Frame 3 (AI Predicted)")

    st.divider()
    # Bottom half: Before and after videos
    st.header("Final Video Comparison")
    
    # We use the same padding logic as the slider to keep portrait videos small
    # [Padding, Original, Predicted, Padding]
    v_pad_l, v_col1, v_col2, v_pad_r = st.columns([0.5, 1, 1, 0.5])
    
    output_vid = "ai_processed_video.mp4"

    with v_col1:
        st.subheader("Original")
        # Wrapping in a container to maintain a fixed-ish scale
        st.video(video_file)
        
    with v_col2:
        st.subheader("With Predicted Frames")
        
        if os.path.exists(output_vid):
            st.video(output_vid)
        else:
            st.info("Click 'Process' to generate AI version...")

    if st.button("Generate AI Video"):
                st.warning("Processing all triplets... this will take a minute.")
                
                # --- FFMPEG LOGIC ---
                temp_dir = "temp_frames"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Clean old frames
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                
                cap = cv2.VideoCapture("temp_video.mp4")
                # Get original metadata for correct reconstruction
                original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                prog_bar = st.progress(0)
                ret1, prev_frame = cap.read()
                count = 0
                
                while True:
                    ret2, curr_frame = cap.read()
                    if not ret2:
                        break
                    
                    # 1. Pre-process for model (Resize to landscape 448x256)
                    f1_in = cv2.resize(prev_frame, (448, 256))
                    f2_in = cv2.resize(curr_frame, (448, 256))
                    
                    t1 = torch.from_numpy(cv2.cvtColor(f1_in, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                    t2 = torch.from_numpy(cv2.cvtColor(f2_in, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                    
                    # 2. Inference
                    inp = torch.cat([t1, t2], dim=0).unsqueeze(0).to('cuda')
                    with torch.no_grad():
                        out = model(inp)
                    
                    # 3. Post-process
                    p_out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    p_out = np.clip(p_out * 255, 0, 255).astype(np.uint8)
                    p_out_bgr = cv2.cvtColor(p_out, cv2.COLOR_RGB2BGR)
                    
                    # --- THE FIX: Resize back to Portrait/Original dimensions ---
                    final_frame = cv2.resize(p_out_bgr, (width_orig, height_orig))
                    
                    # 4. Save frame
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{count:05d}.png"), final_frame)
                    
                    prev_frame = curr_frame
                    count += 1
                    prog_bar.progress(min(count / total_frames, 1.0))

                cap.release()
                
                # 5. Use ffmpeg to stitch
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(original_fps), '-i', f'{temp_dir}/frame_%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_vid
                ]
                subprocess.run(cmd, check=True)
                
                st.success("Video processed successfully!")
                st.rerun()

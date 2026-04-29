import gradio as gr
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from stable_baselines3 import SAC
from isp_pipeline import isp_pipeline
from utils.WBsRGB import rgb_uv_hist

MODEL_PATH = 'rl_awb_stage1_true_final.zip'

print("Loading model...")
model = SAC.load(MODEL_PATH)
print("Model loaded.")

def enhance(image_rgb):
    if image_rgb is None:
        return None, "No image uploaded."
    try:
        img_f32  = image_rgb.astype(np.float32) / 255.0
        img_norm = img_f32 / (img_f32.max() + 1e-6)

        # Build observation — histogram + empty history
        feature_vec = rgb_uv_hist(img_f32).flatten()
        history     = np.zeros(11, dtype=np.float32)
        obs         = np.concatenate([feature_vec, history]).astype(np.float32)

        # Agent picks 5 parameters dynamically per image
        action, _ = model.predict(obs, deterministic=True)
        gamma = 0.5 + (float(action[2]) + 1) / 2 * 1.5
        alpha = 0.8 + (float(action[3]) + 1) / 2 * 1.2
        gain  = 0.5 + (float(action[4]) + 1) / 2 * 1.5

        # White balance via gray world
        mean_r = np.mean(img_norm[:,:,0])
        mean_g = np.mean(img_norm[:,:,1])
        mean_b = np.mean(img_norm[:,:,2])
        gray   = (mean_r + mean_g + mean_b) / 3.0

        wb = img_norm.copy()
        wb[:,:,0] = np.clip(img_norm[:,:,0] * (gray/(mean_r+1e-6)), 0, 1)
        wb[:,:,1] = np.clip(img_norm[:,:,1] * (gray/(mean_g+1e-6)), 0, 1)
        wb[:,:,2] = np.clip(img_norm[:,:,2] * (gray/(mean_b+1e-6)), 0, 1)
        wb = wb / (wb.max() + 1e-6)

        # Apply ISP with agent-selected parameters
        enhanced = isp_pipeline(wb.astype(np.float32), gamma, alpha, gain)
        result   = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)

        params = (
            f"Agent-selected parameters:\n"
            f"  Gamma  (exposure) : {gamma:.3f}\n"
            f"  Alpha  (contrast) : {alpha:.3f}\n"
            f"  Gain (saturation) : {gain:.3f}\n"
        )
        return result, params

    except Exception as e:
        import traceback
        return image_rgb, f"Error: {e}\n{traceback.format_exc()}"

demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type='numpy', label='Upload dark nighttime photo'),
    outputs=[
        gr.Image(label='Enhanced output'),
        gr.Textbox(label='AI-selected parameters', lines=5),
    ],
    title='RL-AWB Extended — Nighttime Image Enhancer',
    description='AI dynamically selects gamma, contrast and saturation per image.'
)

demo.launch(share=False)
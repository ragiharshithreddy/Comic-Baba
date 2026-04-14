#!/usr/bin/env python
"""
scripts/run_ui.py
-----------------
Gradio UI for testing the interpolation model.
"""

import sys
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comic_baba.models.interpolators.rife import RIFEInterpolator

def interpolate_frames(frame1, frame2, factor, model_path):
    # If model_path is empty, use default (init with random weights)
    interpolator = RIFEInterpolator(model_path=model_path if model_path and Path(model_path).exists() else None)

    frames = [np.array(frame1), np.array(frame2)]
    interpolated = interpolator.interpolate(frames, int(factor))

    # Return the middle frame for factor 2
    if len(interpolated) > 2:
        return Image.fromarray(interpolated[1])
    return frame1

def main():
    iface = gr.Interface(
        fn=interpolate_frames,
        inputs=[
            gr.Image(label="Frame 1"),
            gr.Image(label="Frame 2"),
            gr.Slider(minimum=2, maximum=8, step=1, value=2, label="Factor"),
            gr.Textbox(label="Model Checkpoint Path (Optional)")
        ],
        outputs=gr.Image(label="Interpolated Frame"),
        title="Comic-Baba Interpolation Tester",
        description="Upload two consecutive frames to generate an intermediate frame."
    )
    iface.launch()

if __name__ == "__main__":
    main()

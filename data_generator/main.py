from supertuxcart_data_generator import DataGenerator
from supertuxcart_models import load_model
from supertuxcart_visualizer import Visualizer, save_video
import os
import numpy as np

def gen_data(idx, model, visualizer, save_video_param, noise=None):
    os.makedirs(f"../data/{idx}", exist_ok=True)
    data_generator = DataGenerator(model, visualizer=visualizer, noise=noise, save_video=save_video_param)
    data_generator.generate(track_name="lighthouse", max_steps=1500, data_dir=f"../data/{idx}")

    if save_video_param is True:
        frames = visualizer.frames
        save_video(frames, f"video_{idx}.mp4")

def main():
    model = load_model("cnn_planner", with_weights=True)
    visualizer = Visualizer()

    max_noise = (0.1, 5)
    for k in range(30):
        if k == 15:
            gen_data(k, model, visualizer, save_video_param=True, noise=None)
        else:
            gen_data(k, model, visualizer, save_video_param=False, noise=None)

    nm = 0.5
    for k in range(30, 50):
        if k == 60:
            gen_data(k, model, visualizer, save_video_param=True, noise=np.random.randn(2) * max_noise * nm)
        else:
            gen_data(k, model, visualizer, save_video_param=False, noise=np.random.randn(2) * max_noise * nm)

    # nm = 1.0
    # for k in range(90, 150):
    #     if k == 120:
    #         gen_data(k, model, visualizer, save_video_param=True, noise=np.random.randn(2) * max_noise * nm)
    #     else:
    #         gen_data(k, model, visualizer, save_video_param=False, noise=np.random.rand(2) * max_noise * nm)

if __name__ == "__main__":
    main()
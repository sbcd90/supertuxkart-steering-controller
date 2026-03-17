import torch
import numpy as np

from supertuxcart_env import SequentialRollout
from supertuxcart_visualizer import Visualizer


class BasePlanner:
    ALLOWED_INFORMATION = []

    def __init__(self, model: torch.nn.Module, device: str, noise=None):
        self.model = model
        self.model.to(device).eval()
        self.noise = noise

        self.debug_info = {}

    @torch.inference_mode()
    def act(self, batch: dict) -> dict:
        allowed_info = {k: batch.get(k) for k in self.ALLOWED_INFORMATION}
        prediction = self.model(**allowed_info).squeeze(0).cpu().numpy()
        if self.noise is not None:
            prediction += np.random.randn(*prediction.shape) * self.noise[0]

        speed = np.linalg.norm(batch["velocity"].squeeze(0).cpu().numpy())
        if self.noise is not None:
            speed += np.random.randn() * self.noise[1]
        steer, acceleration, brake = self.get_action(prediction, speed)

        return {
            "steer": steer,
            "acceleration": acceleration,
            "brake": brake
        }

    def get_action(self, waypoints: torch.Tensor, speed: torch.Tensor, target_speed: float=5.0,
                   idx: int=2, p_gain: float=10.0, constant_acceleration: float=0.2):
        angle = np.arctan2(waypoints[idx, 0], waypoints[idx, 1])
        steer = p_gain * angle

        acceleration = constant_acceleration if target_speed > speed else 0.0
        brake = False

        self.debug_info.update({"waypoints": waypoints, "steer": steer, "speed": speed})

        steer = float(np.clip(steer, -1, 1))
        acceleration = float(np.clip(acceleration, 0, 1))

        return steer, acceleration, brake

class ImagePlanner(BasePlanner):
    ALLOWED_INFORMATION = ["image"]

class DataGenerator:
    def __init__(self, model: torch.nn.Module, device: str | None=None, visualizer: Visualizer | None=None,
                 noise=None, save_video=False):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_type = model.__class__.__name__
        model_to_planner = {
            "CNNPlanner": ImagePlanner
        }

        if model_type not in model_to_planner:
            raise ValueError(f"Model {model_type} is not supported")

        self.planner = model_to_planner[model_type](model, device=self.device, noise=noise)
        self.visualizer = visualizer
        self.save_video = save_video

    @torch.inference_mode()
    def step(self, state, track, render_data):
        sample = {
            "location": np.float32(state.karts[0].location),
            "front": np.float32(state.karts[0].front),
            "velocity": np.float32(state.karts[0].velocity),
            "distance_down_track": float(state.karts[0].distance_down_track),
            "image_raw": np.uint8(render_data[0].image),
        }
        sample['image'] = np.float32(sample['image_raw']).transpose(2, 0, 1) / 255.0

        batch = torch.utils.data.default_collate([sample])
        batch['distance_down_track'] = batch['distance_down_track'].float()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        action = self.planner.act(batch)
        state = {
            "image": sample["image_raw"],
            "velocity": sample["velocity"],
        }

        if self.visualizer is not None and self.save_video:
            self.visualizer.process(sample, self.planner.debug_info)

        return state, action

    def generate(self, track_name: str="lighthouse", max_steps: int=100, frame_skip: int=4, data_dir="data"):
        rollout = SequentialRollout(
            callback=self.step,
            track_name=track_name,
            max_steps=max_steps,
            frame_skip=frame_skip,
            data_dir=data_dir
        )

        try:
            while True:
                payload = rollout.step()
                if payload is None:
                    break
        finally:
            rollout.close()
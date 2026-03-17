import numpy as np
import torch
from torchvision import transforms

from race import rollout
from visualizations import Visualizer
from supertuxcart_episode_visualizer import bin_center, steer_bins, acceleration_bins

class BasePlanner:
    ALLOWED_INFORMATION = []

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.model.to(device).eval()
        self.device = device

        self.debug_info = None
        self.steer_bins = np.linspace(-1, 1, 11)
        self.acceleration_bins = np.linspace(0, 1, 6)

    @torch.inference_mode()
    def act(self, batch: dict) -> dict:
        allowed_info = {k: batch.get(k).to(self.device) for k in self.ALLOWED_INFORMATION}
        if isinstance(self, DecisionPlanner):
            steer_logits, acceleration_logits, brake_logits = self.model(**allowed_info)
            steer_logits = steer_logits.squeeze(0).detach().cpu().numpy()
            acceleration_logits = acceleration_logits.squeeze(0).detach().cpu().numpy()
            brake_logits = brake_logits.squeeze(0).detach().cpu().numpy()

            steer = np.argmax(steer_logits[-1])
            acceleration = np.argmax(acceleration_logits[-1])
            brake = np.argmax(brake_logits[-1])
        else:
            steer = self.model(**allowed_info)

        speed = np.linalg.norm(batch["current_velocity"])
        constant_acceleration = 0.2
        target_speed = 5.0
        if isinstance(self, DecisionPlanner):
            steer, acceleration, brake = (bin_center(steer, steer_bins),
                                          constant_acceleration if target_speed > speed else 0.0, False)
        else:
            steer, acceleration, brake = (steer.squeeze().detach().cpu().numpy(),
                                          constant_acceleration if target_speed > speed else 0.0, False)

        steer = float(np.clip(steer, -1, 1))
        acceleration = float(np.clip(acceleration, 0, 1))
        return {
            "steer": steer,
            "acceleration": acceleration,
            "brake": brake
        }

class DecisionPlanner(BasePlanner):
    ALLOWED_INFORMATION = ["images", "velocities", "actions", "reward_to_gos"]

class CNNPlanner(BasePlanner):
    ALLOWED_INFORMATION = ["image"]


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: str | None=None, visualizer: Visualizer | None=None):
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
            "CNNPlanner": CNNPlanner,
        }

        if model_type not in model_to_planner:
            raise ValueError(f"Unknown model type: {model_type}")

        self.planner = model_to_planner[model_type](model, device=self.device)
        self.visualizer = visualizer

        self.images = []
        self.velocities = []
        self.prev_actions = []
        self.return_to_gos = []

        self.mean_rtg = 167.0331
        self.std_rtg = 260.4325
        self.max_rtg = 900
        self.prev_actions.append([0.0, 0.0, 0.0])
        self.context_len = 20

        self.transform = transforms.ToTensor()

    @torch.inference_mode()
    def step(self, state, render_data):
        sample = {
            "location": np.float32(state.karts[0].location),
            "front": np.float32(state.karts[0].front),
            "velocity": np.float32(state.karts[0].velocity),
            "distance_down_track": float(state.karts[0].distance_down_track),
            "image_raw": np.uint8(render_data[0].image),
            "return_to_go": self.max_rtg - float(state.karts[0].distance_down_track)
        }

        sample["image"] = self.transform(sample["image_raw"])[None]
        self.images.append(sample["image"])
        self.velocities.append(sample["velocity"])
        self.return_to_gos.append((sample["return_to_go"] - self.mean_rtg) / self.std_rtg)

        images_tensor = torch.cat(self.images)
        prev_actions_tensor = torch.as_tensor(self.prev_actions, dtype=torch.float32)
        reward_to_gos_tensor = torch.as_tensor(self.return_to_gos, dtype=torch.float32)
        velocities_tensor = torch.as_tensor(np.array(self.velocities), dtype=torch.float32)

        idx = len(self.images) - 1
        start = max(0, idx - self.context_len + 1)
        end = idx + 1

        images_seq = images_tensor[start:end]
        velocities_seq = velocities_tensor[start:end]
        prev_actions_seq = prev_actions_tensor[start:end]
        reward_to_gos_seq = reward_to_gos_tensor[start:end]

        pad = self.context_len - images_seq.shape[0]
        if pad > 0:
            images_seq = torch.cat(
                [torch.zeros(pad, *images_seq.shape[1:]), images_seq],
                dim=0
            )

            velocities_seq = torch.cat(
                [torch.zeros(pad, velocities_seq.shape[1]), velocities_seq],
                dim=0
            )

            prev_actions_seq = torch.cat(
                [torch.zeros(pad, prev_actions_seq.shape[1]), prev_actions_seq],
                dim=0
            )

            reward_to_gos_seq = torch.cat(
                [torch.zeros(pad), reward_to_gos_seq],
                dim=0
            )
        images_seq = images_seq.unsqueeze(0)
        velocities_seq = velocities_seq.unsqueeze(0)
        prev_actions_seq = prev_actions_seq.unsqueeze(0)
        reward_to_gos_seq = reward_to_gos_seq.unsqueeze(0)

        batch = {
            "images": images_seq,
            "velocities": velocities_seq,
            "actions": prev_actions_seq,
            "reward_to_gos": reward_to_gos_seq,
            "current_velocity": sample["velocity"],
            "image": sample["image"]
        }
        action = self.planner.act(batch)
        self.prev_actions.append([action["steer"], action["acceleration"], action["brake"]])

        if self.visualizer is not None:
            self.visualizer.process(sample)

        return action

    def evaluate(
        self,
        track_name: str = "lighthouse",
        max_steps: int = 100,
        frame_skip: int = 4,
        disable_progress: bool = False,
    ):
        max_distance = 0.0
        total_track_distance = float("inf")

        with rollout(
            callback=self.step,
            track_name=track_name,
            max_steps=max_steps,
            frame_skip=frame_skip,
            disable_progress=disable_progress,
        ) as rollout_loop:
            for i, payload in enumerate(rollout_loop):
                state = payload["state"]

                # update how far the kart has gone
                max_distance = max(max_distance, state.karts[0].distance_down_track)

        return max_distance, total_track_distance
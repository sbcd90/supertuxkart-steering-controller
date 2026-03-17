import pathlib

import numpy as np
import torch
from torchvision import transforms

from data_generator.supertuxcart_serde import load_dict
from data_generator.supertuxcart_visualizer import save_video

steer_bins = np.linspace(-1, 1, 11)
acceleration_bins = np.linspace(0, 1, 6)
brake_bins = [0, 1]

def bin_idx(x, bins):
    idx = np.digitize(x, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    return idx


def bin_center(idx, bins):
    left = bins[idx]
    right = bins[idx + 1]
    return (left + right) / 2

class LazySuperTuxLoader:
    def __init__(self, path="data", transform=None):
        self.episodes = list(set([k.parent for k in pathlib.Path(path).rglob("*.pt")]))
        print(f"Number of episodes: {len(self.episodes)}")
        # self.visualizer = Visualizer()

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor(),
            ])

    def num_episodes(self):
        return len(self.episodes)

    def load_episode(self, idx, is_bin_enabled=True):
        path_idx = self.episodes[idx]

        images = []
        velocities = []
        actions = []
        rewards = []
        timesteps = []

        for path in pathlib.Path(path_idx).rglob("*.pt"):
            if not (path.name.split(".")[0]).isnumeric():
                continue

            d = load_dict(path)
            image = d["image"]
            images.append(self.transform(image)[None])

            velocity = d["velocity"]
            velocities.append(velocity)

            steer = d["steer"]
            acceleration = d["acceleration"]
            brake = d["brake"]
            if is_bin_enabled:
                action = [bin_idx(steer, steer_bins),
                          bin_idx(acceleration, acceleration_bins),
                          (0 if brake == 0.0 else 1)]
            else:
                action = [steer, acceleration, brake]
            actions.append(action)

            reward = d["reward"]
            rewards.append(reward)

            timesteps.append(int(path.name.split(".")[0]))

        images_tensor = torch.cat(images)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        velocities_tensor = torch.as_tensor(np.array(velocities), dtype=torch.float32)
        timesteps_tensor = torch.as_tensor(timesteps, dtype=torch.float32)

        order = torch.argsort(timesteps_tensor)
        timesteps_tensor = timesteps_tensor[order]
        images_tensor = images_tensor[order]
        actions_tensor = actions_tensor[order]
        velocities_tensor = velocities_tensor[order]
        rewards_tensor = rewards_tensor[order]

        rewards_cumulative = rewards_tensor.cumsum(dim=0)
        rewards_to_go_tensor = rewards_cumulative[-1] - rewards_cumulative

        prev_actions_tensor = torch.zeros_like(actions_tensor)
        prev_actions_tensor[1:] = actions_tensor[:-1]

        return (
            timesteps_tensor,
            images_tensor,
            actions_tensor,
            prev_actions_tensor,
            velocities_tensor,
            rewards_to_go_tensor
        )

    def save_video(self, images, filename="video.mp4"):
        for image in images:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()
                image = image.permute(1, 2, 0)
                image = image.numpy()
            image = (image * 255).clip(0, 255).astype("uint8")
            sample = {
                "image_raw": image
            }
            self.visualizer.process(sample)
        frames = self.visualizer.frames
        save_video(frames, filename)

    def load_all_episodes(self, num_episodes: int, is_bin_enabled=True):
        episodes = []
        for idx in range(num_episodes):
            episode = self.load_episode(idx, is_bin_enabled)
            episodes.append(episode)
        return episodes

    def get_reward_to_gos_mean_std_max(self):
        episodes = self.load_all_episodes(self.num_episodes())

        all_return_to_gos = []
        for episode in episodes:
            return_to_gos = episode[-1]
            all_return_to_gos.append(return_to_gos)
        all_return_to_gos = torch.cat(all_return_to_gos)

        mean = all_return_to_gos.mean()
        std = all_return_to_gos.std() + 1e-6
        max_return = all_return_to_gos.max()
        return mean, std, max_return

if __name__ == "__main__":
    loader = LazySuperTuxLoader("data")

    print(f"Episodes: {loader.num_episodes()}")
    episodes = loader.load_all_episodes(loader.num_episodes(), False)
    mean_rtg, std_rtg, max_rtg = loader.get_reward_to_gos_mean_std_max()
    timesteps_tensor, images_tensor, _, _, _, _ = loader.load_episode(0)
    timesteps_tensor1, images_tensor1, _, _, _, _ = loader.load_episode(0)

    loader.save_video(images_tensor1, "decrypt_video.mp4")
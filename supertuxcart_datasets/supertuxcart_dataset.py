import torch
from torch.utils.data import Dataset

class DecisionDataset(Dataset):
    def __init__(self, episodes, context_len=20):
        self.context_len = context_len
        self.indices = []

        all_return_to_gos = []
        for episode in episodes:
            return_to_gos = episode[-1]
            all_return_to_gos.append(return_to_gos)
        all_return_to_gos = torch.cat(all_return_to_gos)

        mean = all_return_to_gos.mean()
        std = all_return_to_gos.std() + 1e-6

        normalized_episodes = []
        for episode in episodes:
            timesteps, images, actions, prev_actions, velocities, rtgs = episode
            rtgs = (rtgs - mean) / std

            normalized_episodes.append(
                (timesteps, images, actions, prev_actions, velocities, rtgs)
            )
        self.episodes = normalized_episodes

        for idx, episode in enumerate(self.episodes):
            timesteps = episode[0]
            episode_len = timesteps.shape[0]

            for i in range(episode_len):
                start = max(0, i - context_len + 1)
                end = i + 1
                self.indices.append((idx, start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx, start, end = self.indices[idx]

        timesteps, images, actions, prev_actions, velocities, return_to_gos = self.episodes[idx]

        images_seq = images[start:end]
        velocities_seq = velocities[start:end]
        actions_seq = actions[start:end]
        prev_actions_seq = prev_actions[start:end]
        return_to_gos_seq = return_to_gos[start:end]

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

            actions_seq = torch.cat(
                [torch.zeros(pad, actions_seq.shape[1]), actions_seq],
                dim=0
            )

            prev_actions_seq = torch.cat(
                [torch.zeros(pad, prev_actions_seq.shape[1]), prev_actions_seq],
                dim=0
            )

            return_to_gos_seq = torch.cat(
                [torch.zeros(pad), return_to_gos_seq],
                dim=0
            )
        return images_seq, velocities_seq, actions_seq, prev_actions_seq, return_to_gos_seq

class CNNDataset(Dataset):
    def __init__(self, episodes):
        self.all_data_points = []
        for episode in episodes:
            _, images, actions, _, _, _ = episode
            for i, (image, action) in enumerate(zip(images, actions)):
                self.all_data_points.append([image, action[0]])
        print()

    def __len__(self):
        return len(self.all_data_points)

    def __getitem__(self, idx):
        return self.all_data_points[idx]
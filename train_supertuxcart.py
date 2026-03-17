import argparse

import torch
import numpy as np

from models import load_model, save_model
from supertuxcart_datasets.supertuxcart_dataset import DecisionDataset
from supertuxcart_episode_visualizer import LazySuperTuxLoader


def train(
    model_name: str="supertuxcart_decision_transformer",
    num_epoch: int=50,
    lr: float=1e-4,
    batch_size: int=32,
    seed: int=2026,
    weight_decay: float=None,
    train: bool=True,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    supertuxcart_loader = LazySuperTuxLoader()
    episodes = supertuxcart_loader.load_all_episodes(supertuxcart_loader.num_episodes())

    num_episodes = len(episodes)
    train_size = int(num_episodes * 0.8)
    val_size = num_episodes - train_size

    perm = torch.randperm(num_episodes)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_episodes = [episodes[i] for i in train_idx]
    val_episodes = [episodes[i] for i in val_idx]

    train_dataset = DecisionDataset(train_episodes)
    val_dataset = DecisionDataset(val_episodes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False)
    model.to(device)
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    if weight_decay is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": []}
        for images, velocities, actions, prev_actions, return_to_gos in train_loader:
            images = images.to(device)
            velocities = velocities.to(device)
            actions = actions.to(device)
            prev_actions = prev_actions.to(device)
            return_to_gos = return_to_gos.to(device)

            steer_logits, acceleration_logits, brake_logits = model(images, velocities, prev_actions, return_to_gos)

            steer_actions = actions[:, :, 0].reshape(-1).long()
            loss_steer = loss_func(steer_logits.reshape(-1, 10), steer_actions)

            acceleration_actions = actions[:, :, 1].reshape(-1).long()
            loss_acceleration = loss_func(acceleration_logits.reshape(-1, 5), acceleration_actions)

            brake_actions = actions[:, :, 2].reshape(-1).long()
            loss_brake = loss_func(brake_logits.reshape(-1, 2), brake_actions)
            loss = loss_steer + loss_acceleration + loss_brake

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted_steer = torch.max(steer_logits.reshape(-1, 10), 1)
            _, predicted_acceleration = torch.max(acceleration_logits.reshape(-1, 5), 1)
            _, predicted_brake = torch.max(brake_logits.reshape(-1, 2), 1)
            predicted_action = torch.cat([predicted_steer.unsqueeze(-1), predicted_acceleration.unsqueeze(-1),
                                          predicted_brake.unsqueeze(-1)], 1)
            matched = torch.sum(torch.eq(predicted_action, actions.reshape(-1, 3)))

            metrics["train_acc"].append(matched.item() / len(actions))
        with torch.inference_mode():
            for images, velocities, actions, prev_actions, return_to_gos in val_loader:
                images = images.to(device)
                velocities = velocities.to(device)
                actions = actions.to(device)
                prev_actions = prev_actions.to(device)
                return_to_gos = return_to_gos.to(device)

                steer_logits, acceleration_logits, brake_logits = model(images, velocities, prev_actions, return_to_gos)
                _, predicted_steer = torch.max(steer_logits.reshape(-1, 10), 1)
                _, predicted_acceleration = torch.max(acceleration_logits.reshape(-1, 5), 1)
                _, predicted_brake = torch.max(brake_logits.reshape(-1, 2), 1)
                predicted_action = torch.cat([predicted_steer.unsqueeze(-1), predicted_acceleration.unsqueeze(-1),
                                              predicted_brake.unsqueeze(-1)], 1)
                matched = torch.sum(torch.eq(predicted_action, actions.reshape(-1, 3)))

                metrics["val_acc"].append(matched.item() / len(actions))

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_acc:.4f} "
            f"val_loss={epoch_val_acc:.4f}"
        )
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        pass

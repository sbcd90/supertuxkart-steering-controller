import argparse

import numpy as np
import torch

from models import load_model, RegressionLoss, save_model
from supertuxcart_datasets.supertuxcart_dataset import CNNDataset
from supertuxcart_episode_visualizer import LazySuperTuxLoader


def train(
    model_name: str="supertuxcart_cnn",
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
    episodes = supertuxcart_loader.load_all_episodes(supertuxcart_loader.num_episodes(),
                                                     is_bin_enabled=False)

    num_episodes = len(episodes)
    train_size = int(num_episodes * 0.8)

    perm = torch.randperm(num_episodes)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_episodes = [episodes[i] for i in train_idx]
    val_episodes = [episodes[i] for i in val_idx]

    train_dataset = CNNDataset(train_episodes)
    val_dataset = CNNDataset(val_episodes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, with_weights=False)
    model.to(device)
    model.train()

    loss_func = RegressionLoss()
    if weight_decay is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        metrics = {"train_loss": 0.0, "val_loss": 0.0}
        for images, steers in train_loader:
            images, steers = images.to(device), steers.to(device)
            actual_steers = steers.unsqueeze(-1)

            predicted_steers = model(images)

            optimizer.zero_grad()
            loss = loss_func(predicted_steers, actual_steers)
            loss.backward()
            optimizer.step()

            metrics["train_loss"] += loss.item()
        with torch.inference_mode():
            for images, steers in val_loader:
                images, steers = images.to(device), steers.to(device)
                actual_steers = steers.unsqueeze(-1)

                predicted_steers = model(images)
                loss = loss_func(predicted_steers, actual_steers)

                metrics["val_loss"] += loss.item()

        epoch_train_loss = metrics["train_loss"] / len(train_loader)
        epoch_val_loss = metrics["val_loss"] / len(val_loader)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
        )
    save_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train", type=bool, default=False)

    args = vars(parser.parse_args())
    if args["train"]:
        train(**args)
    else:
        pass
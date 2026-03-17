from pathlib import Path

import torch
import torch.nn as nn

class CNNPlanner(torch.nn.Module):
    def __init__(self, n_waypoints=3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.flatten_size = 128 * (96 // 8) * (128 // 8)

        self.linear1 = nn.Linear(self.flatten_size, 128)
        self.linear2 = nn.Linear(128, self.n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(image))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_size)

        x = nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        x = x.view(-1, self.n_waypoints, 2)
        return x

MODEL_FACTORY = {
    "cnn_planner": CNNPlanner,
}

def load_model(
        model_name: str = "cnn_planner",
        with_weights:bool = True,
        **model_kwargs
) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m

def save_model(model: nn.Module) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = Path(__file__).resolve().parent / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path
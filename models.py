from pathlib import Path

import torch
from torch import nn


class StateEncoder(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.image_fc = nn.Linear(128, embed_dim)
        self.velocity_fc = nn.Linear(3, embed_dim)

    def forward(self, image, velocity):

        x = self.cnn(image)
        x = x.flatten(1)  # faster and safer than view

        img = self.image_fc(x)
        vel = self.velocity_fc(velocity)

        return img + vel

class DecisionTransformer(nn.Module):

    def __init__(self, embed_dim=128, n_heads=4, n_layers=4):
        super().__init__()

        self.embed_dim = embed_dim

        self.state_encoder = StateEncoder(embed_dim)

        self.action_embed = nn.Linear(3, embed_dim)
        self.reward_to_go_embed = nn.Linear(1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.steer_head = nn.Linear(embed_dim, 10)
        self.acceleration_head = nn.Linear(embed_dim, 5)
        self.brake_head = nn.Linear(embed_dim, 2)

    def forward(self, images, velocities, actions, reward_to_gos):

        B, T = actions.shape[:2]

        images_flat = images.reshape(B * T, *images.shape[2:])
        velocities_flat = velocities.reshape(B * T, velocities.shape[-1])

        state_tokens = self.state_encoder(images_flat, velocities_flat)
        state_tokens = state_tokens.reshape(B, T, self.embed_dim)

        action_tokens = self.action_embed(actions)

        reward_to_go_tokens = self.reward_to_go_embed(
            reward_to_gos.unsqueeze(-1)
        )

        tokens = state_tokens + action_tokens + reward_to_go_tokens

        h = self.transformer(tokens)

        steer_logits = self.steer_head(h)
        acceleration_logits = self.acceleration_head(h)
        brake_logits = self.brake_head(h)

        return steer_logits, acceleration_logits, brake_logits

class RegressionLoss(nn.Module):
    def forward(self, raw_steer: torch.Tensor, actual_steer: torch.Tensor):
        return nn.functional.mse_loss(raw_steer, actual_steer)

class CNNPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.flatten_size = 128 * (96 // 8) * (128 // 8)

        self.linear1 = nn.Linear(self.flatten_size, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        #raise NotImplementedError
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

        x = x.view(-1, 1)
        return x

MODEL_FACTORY = {
    "supertuxcart_decision_transformer": DecisionTransformer,
    "supertuxcart_cnn": CNNPlanner,
}

def load_model(
        model_name: str = "supertuxcart_decision_transformer",
        with_weights:bool = False,
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

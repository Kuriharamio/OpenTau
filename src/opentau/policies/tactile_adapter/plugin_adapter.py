import torch
from torch import Tensor, nn


def _pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    new_vector[..., :current_dim] = vector
    return new_vector


class PluginTactileAdapter(nn.Module):
    """Plugin tactile adapter with dual-path encoding and gated fusion."""

    def __init__(self, input_dim: int, proj_dim: int, num_tokens: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_tokens = num_tokens

        self.num_fingers = 3
        self.num_points = 52
        self.per_point_dim = 6
        self.summary_input_dim = self.num_fingers * self.per_point_dim
        self.detail_input_dim = self.num_fingers * self.num_points * self.per_point_dim

        self.finger_summary_proj = nn.Linear(self.summary_input_dim, proj_dim)
        self.point_detail_proj = nn.Linear(self.detail_input_dim, proj_dim)
        self.gate_proj = nn.Linear(proj_dim * 2, proj_dim * 2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.token_positional = nn.Parameter(torch.zeros(num_tokens, proj_dim))

    def _reshape_tactile(self, tactile: Tensor) -> Tensor:
        if tactile.ndim == 3:
            tactile = tactile.unsqueeze(0)

        if tactile.ndim == 2:
            if tactile.shape[-1] < self.detail_input_dim:
                tactile = _pad_vector(tactile, self.detail_input_dim)
            elif tactile.shape[-1] > self.detail_input_dim:
                tactile = tactile[..., : self.detail_input_dim]
            tactile = tactile.reshape(-1, self.num_fingers, self.num_points, self.per_point_dim)
            return tactile

        if tactile.ndim != 4:
            raise ValueError(f"Unsupported tactile tensor shape {tuple(tactile.shape)}")

        batch_size = tactile.shape[0]
        flattened = tactile.reshape(batch_size, -1)
        if flattened.shape[-1] < self.detail_input_dim:
            flattened = _pad_vector(flattened, self.detail_input_dim)
        elif flattened.shape[-1] > self.detail_input_dim:
            flattened = flattened[..., : self.detail_input_dim]

        return flattened.reshape(batch_size, self.num_fingers, self.num_points, self.per_point_dim)

    def forward(self, tactile: Tensor) -> Tensor:
        proj_dtype = self.finger_summary_proj.weight.dtype
        tactile = self._reshape_tactile(tactile).to(dtype=proj_dtype)
        batch_size = tactile.shape[0]

        finger_summary = tactile.mean(dim=2).reshape(batch_size, -1)
        path_a = self.finger_summary_proj(finger_summary)

        point_detail = tactile.reshape(batch_size, -1)
        path_b = self.point_detail_proj(point_detail)

        combined = torch.cat([path_a, path_b], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        gated = gate * combined

        sinusoid = torch.sin(gated) + torch.cos(gated)
        tactile_embedding = self.fusion_mlp(sinusoid)

        tactile_tokens = tactile_embedding[:, None, :].expand(-1, self.num_tokens, -1)
        tactile_tokens = tactile_tokens + self.token_positional[None, :, :].to(dtype=tactile_tokens.dtype)
        return tactile_tokens

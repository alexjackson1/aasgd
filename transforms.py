from beartype import beartype
from beartype.typing import List, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


@beartype
class ForCredulousInference(T.BaseTransform):
    def __init__(self, semantics: str):
        assert semantics in ["GR", "CO", "ST", "SST", "STG", "ID", "PR"]
        self.semantics = semantics

    def forward(self, data: Data):
        extensions: Optional[torch.Tensor] = data["extensions"][self.semantics]
        if extensions is None:
            raise ValueError(f"{self.semantics} extensions are required.")

        ext = (extensions.sum(dim=0) > 0).long()
        data["y"] = ext.float().unsqueeze(-1)
        return data


@beartype
class ForSkepticalInference(T.BaseTransform):
    def __init__(self, semantics: str):
        assert semantics in ["GR", "CO", "ST", "SST", "STG", "ID", "PR"]
        self.semantics = semantics

    def forward(self, data: Data):
        extensions: Optional[torch.Tensor] = data["extensions"][self.semantics]
        if extensions is None:
            raise ValueError(f"{self.semantics} extensions are required.")

        ext = extensions.sum(dim=0) == len(extensions)
        data["y"] = ext.float().unsqueeze(-1)
        return data


@beartype
class AddGroundedFeature(T.BaseTransform):
    def forward(self, data: Data):
        extensions: Optional[torch.Tensor] = data["extensions"]["GR"]
        if extensions is None:
            raise ValueError("Grounded extensions are required.")

        data.x = torch.cat([data.x, extensions.t()], dim=1)
        return data


if __name__ == "__main__":
    transform = T.Compose([AddGroundedFeature(), ForSkepticalInference("GR")])
    dataset: List[dict] = torch.load("data.pt", weights_only=False, map_location="cpu")
    dataset: List[Data] = [Data(**data) for data in dataset]

    for data in dataset:
        data = transform(data)
        print(data)
        break

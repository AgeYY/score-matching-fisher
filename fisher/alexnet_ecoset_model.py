"""Shared AlexNet-EcoSet model loader."""

from __future__ import annotations

import torch


ALEXNET_ECOSET_URL = "https://osf.io/t6h3c/download"
ALEXNET_ECOSET_FILE_NAME = "Alexnet_ecoset"
ALEXNET_ECOSET_NUM_CLASSES = 565


def load_alexnet_ecoset(device: str | torch.device) -> tuple[torch.nn.Module, object]:
    """Load AlexNet with the EcoSet-trained OSF weights and preprocessing."""
    try:
        from torchvision import models
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision to load AlexNet-EcoSet.") from exc

    dev = torch.device(device)
    model = models.alexnet(weights=None, num_classes=ALEXNET_ECOSET_NUM_CLASSES)
    state_dict = torch.hub.load_state_dict_from_url(
        ALEXNET_ECOSET_URL,
        map_location=dev,
        file_name=ALEXNET_ECOSET_FILE_NAME,
    )
    model.load_state_dict(state_dict)
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model.to(dev).eval(), transform

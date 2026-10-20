"""Six-channel image encoders and self-supervised objectives."""

from importlib import import_module

from .mae import MAEConfig, MaskedAutoencoder
from .cnn_mae import CNNMaskedAutoencoder

# import_module supports the exact user-requested CA-MAE.py filename.
ChannelAgnosticMAE = import_module(".CA-MAE", __name__).ChannelAgnosticMAE
DiffMAE = import_module(".DiffMAE", __name__).DiffMAE

CHECKPOINT_FORMATS = {"cellvit-spatial-mae-v1", "cellvit-masked-autoencoder-v2"}


def build_model(config: MAEConfig | None = None) -> MaskedAutoencoder:
    """Shared dispatch for training, reconstruction, and embedding export."""
    config = config or MAEConfig()
    classes = {
        "mae": MaskedAutoencoder, "ca_mae": ChannelAgnosticMAE,
        "diffmae": DiffMAE, "cnn_mae": CNNMaskedAutoencoder,
    }
    return classes[config.model_type](config)


def model_from_checkpoint(state: dict) -> MaskedAutoencoder:
    """Load legacy spatial MAEs and new architecture-tagged checkpoints."""
    if state.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError("Not a supported CellViT masked autoencoder checkpoint")
    config = MAEConfig(**state["model_config"])
    if state["format"] == "cellvit-spatial-mae-v1" and config.model_type != "mae":
        raise ValueError("Legacy spatial checkpoints must contain a spatial MAE")
    model = build_model(config)
    model.load_state_dict(state["model"])
    return model

__all__ = [
    "MAEConfig", "MaskedAutoencoder", "ChannelAgnosticMAE", "DiffMAE",
    "CNNMaskedAutoencoder", "build_model", "model_from_checkpoint", "CHECKPOINT_FORMATS",
]

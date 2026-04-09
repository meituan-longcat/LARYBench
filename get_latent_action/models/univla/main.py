from genie.dataset import LightningOpenX
from genie.model import DINO_LAM
from lightning.pytorch.cli import LightningCLI

cli = LightningCLI(
    DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
)

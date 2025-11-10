from dataclasses import dataclass
from typing import Optional, Literal

Label = Literal["Falsifiable", "NonFalsifiable"]


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    source: str
    veracity: Optional[str]
    domain: Optional[str]     # "food", "fitness", "general"
    falsifiability: Optional[Label]

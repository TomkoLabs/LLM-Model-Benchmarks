from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionPolicy:
    max_member_length: int = 255
    allow_overwrite: bool = False

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import numpy as np

# @dataclass
# class DitString:
#     length: int
#     dimension: int
#     dit_string: list[int]

#     def __init__(
#         self,
#         dit_string: Iterable[int],
#         length: int | None = None,
#         dimension: int = 2,
#     ) -> None:
#         values = [int(v) for v in dit_string]
#         self.length = len(values) if length is None else int(length)
#         self.dimension = int(dimension)
#         if self.length != len(values):
#             raise ValueError("length must match the provided dit_string size.")
#         if self.dimension < 2:
#             raise ValueError("dimension must be >= 2.")
#         for value in values:
#             if value < 0 or value >= self.dimension:
#                 raise ValueError("Each dit value must be in [0, dimension - 1].")
#         self.dit_string = values

#     def __iter__(self):
#         return iter(self.dit_string)

#     def __len__(self) -> int:
#         return self.length

#     def __getitem__(self, index):
#         return self.dit_string[index]

#     def __array__(self, dtype=None):
#         return np.asarray(self.dit_string, dtype=dtype)

#     def tolist(self) -> list[int]:
#         return list(self.dit_string)

@dataclass
class Sample:
    indexes: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dit_strings: list = field(default_factory=list)


@dataclass
class Restriction:
    dit_restrictions: list[int] | None = None
    dit_value_restrictions: list[int] | None = None
    additional_dits_val: int = 0

    def __init__(
        self,
        dit_restrictions: list[int] | None = None,
        dit_value_restrictions: list[int] | None = None,
        additional_dits_val: int = 0,
    ) -> None:
        self.dit_restrictions = dit_restrictions
        self.dit_value_restrictions = dit_value_restrictions
        self.additional_dits_val = additional_dits_val

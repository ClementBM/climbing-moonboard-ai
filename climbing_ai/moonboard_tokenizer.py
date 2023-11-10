import string
from enum import Enum, unique
from typing import List, NamedTuple
from math import hypot


@unique
class HoldType(Enum):
    NONE = 0
    FOOT = 1
    HAND = 2
    BOTH = 3


class Position(NamedTuple):
    x: int
    y: int

    def __sub__(self, other):
        d_x = other.x - self.x
        d_y = other.y - self.y
        return hypot(d_x, d_y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Hold:
    def __init__(self, name: str, id: int, location: Position):
        self.name = name
        self.location = location
        self.id = id

    def __repr__(self):
        return f"{self.name}:{self.id} {self.position}"


class MoonboardEncodings:
    def __init__(self, ids, locations, holds):
        self.ids = ids
        self.holds = holds
        self.locations = locations

    def order_hold_sequence(self):
        holds = [
            Hold(name, id, location)
            for name, id, location in zip(self.holds, self.ids, self.locations)
        ]

        holds.sort(key=lambda x: x.id)

        self.ids = [hold.id for hold in holds]
        self.locations = [hold.location for hold in holds]
        self.holds = [hold.name for hold in holds]


SPECIAL_POSITION = Position(x=-100, y=-100)


class MoonboardTokenizer:
    def __init__(
        self,
        horizontal_count: int,
        vertical_count: int,
        horizontal_spacing: float,
        vertical_spacing: float,
    ) -> None:
        self._enable_padding = False

        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        self.horizontal_count = horizontal_count
        self.vertical_count = vertical_count

        self.alpha_indices = list(string.ascii_uppercase[:horizontal_count])
        self.number_indices = list(range(1, vertical_count + 1))

        self.hold_dictionary = self.get_hold_dictionary(
            alpha_indices=self.alpha_indices,
            number_indices=self.number_indices,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
        self.sob_token = "[CLS]"
        self.sob_token_id = 0
        self.mask_token = "[MASK]"
        self.mask_token_id = len(self.hold_dictionary) + 1
        self.eob_token = "[EOB]"
        self.eob_token_id = len(self.hold_dictionary) + 2
        self.pad_token = "[PAD]"
        self.pad_token_id = len(self.hold_dictionary) + 3

        self.hold_dictionary.update(
            {
                self.sob_token: Hold(
                    name=self.sob_token, location=SPECIAL_POSITION, id=self.sob_token_id
                ),
                self.eob_token: Hold(
                    name=self.eob_token, location=SPECIAL_POSITION, id=self.eob_token_id
                ),
                self.pad_token: Hold(
                    name=self.pad_token, location=SPECIAL_POSITION, id=self.pad_token_id
                ),
                self.mask_token: Hold(
                    name=self.mask_token,
                    location=SPECIAL_POSITION,
                    id=self.mask_token_id,
                ),
            }
        )

        self.decode_dictionary = {v.id: k for k, v in self.hold_dictionary.items()}

        self.special_tokens = [
            self.sob_token,
            self.mask_token,
            self.eob_token,
            self.pad_token,
        ]

    def get_hold_dictionary(
        self,
        alpha_indices: list,
        number_indices: list,
        horizontal_spacing: float,
        vertical_spacing: float,
        id_offset=1,
    ):
        hold_dict = {}
        for alpha_index, alpha_name in enumerate(alpha_indices):
            for number_index, number_name in enumerate(number_indices):
                hold_name = alpha_name + str(number_name)

                hold_location = Position(
                    x=int(alpha_index * horizontal_spacing),
                    y=int(number_index * vertical_spacing),
                )

                hold_id = id_offset + (alpha_index * 18) + number_index

                hold_dict[hold_name] = Hold(
                    name=hold_name, location=hold_location, id=hold_id
                )

        return hold_dict

    def encode(self, sequence_holds) -> MoonboardEncodings:
        holds_ids = []
        hold_locations = []

        if self._enable_padding:
            sequence_holds += [self.pad_token] * (self.max_len - len(sequence_holds))

        for hold_name in sequence_holds:
            hold: Hold = self.hold_dictionary[hold_name]
            holds_ids.append(hold.id)
            hold_locations.append(hold.location)

        return MoonboardEncodings(
            ids=holds_ids,
            locations=hold_locations,
            holds=sequence_holds,
        )

    def decode(self, sequence_ids):
        holds = []
        for id in sequence_ids:
            holds.append(self.decode_dictionary[id])
        return holds

    def get_vocab_size(self, with_added_tokens=True):
        if with_added_tokens:
            return len(self.decode_dictionary)
        else:
            return len(self.decode_dictionary) - len(self.special_tokens)

    def enable_padding(
        self,
        length,
    ):
        self._enable_padding = True
        self.max_len = length

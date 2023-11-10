import torch
import torch.nn as nn
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from artifacts import DATA_FOLDER
import json
import numpy as np
from random import randint, random

from climbing_ai.moonboard_tokenizer import SPECIAL_POSITION, MoonboardTokenizer


def data_preprocessing():
    data_files = [f for f in listdir(DATA_FOLDER) if isfile(DATA_FOLDER / f)]
    dataset = []

    for data_file in data_files:
        boulder_path = DATA_FOLDER / data_file
        grade = data_file.replace(".json", "")

        with open(boulder_path) as json_data:
            json_raw = json_data.read()
            boulder_problems = json.loads(json_raw)

        for boulder_problem in boulder_problems["Data"]:
            moves = boulder_problem["Moves"]
            hold_locations = boulder_problem["Locations"]

            if len(moves) != len(hold_locations):
                print("move count != hold count")

            holds = []
            locations = []
            for j in range(0, len(moves)):
                holds.append(moves[j]["Description"])
                locations.append((hold_locations[j]["X"], hold_locations[j]["Y"]))

            dataset.append(
                {
                    "holds": holds,
                    "locations": locations,
                    "grade": grade,
                }
            )
    return dataset


class MoonboardDataset(Dataset):
    # used for grade serialization as an integer ordinal scale
    grades = ["6A+", "6B", "6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+"]

    def __init__(
        self,
        dataset,
        tokenizer: MoonboardTokenizer,
        max_len,
        denoising_ratio=0.15,
        ignore_index=-100,
    ):
        super().__init__()

        self.max_len = max_len
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.denoising_ratio = denoising_ratio

        self.vocab_size = self.tokenizer.get_vocab_size(False)
        self.max_pred_count = self.max_len - 2
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        boulder = self.dataset[idx]

        # Transform the holds into tokens
        encodings = self.tokenizer.encode(boulder["holds"])
        # Order holds from left to right, bottom to top
        encodings.order_hold_sequence()

        # Add sob, eob and padding to each sentence
        num_padding_tokens = self.max_len - len(encodings.ids) - 2

        # Make sure the number of padding tokens is not negative
        if num_padding_tokens < 0:
            raise ValueError("Boulder is too long")

        # input token ids
        input_ids = (
            [self.tokenizer.sob_token_id]
            + encodings.ids
            + [self.tokenizer.eob_token_id]
            + [self.tokenizer.pad_token_id] * num_padding_tokens
        )

        # input spatial locations
        special_position = (SPECIAL_POSITION.x, SPECIAL_POSITION.y)
        input_locations = np.array(
            [special_position]
            + encodings.locations
            + [special_position] * (num_padding_tokens + 1)
        )

        grade = self.grades.index(boulder["grade"])

        # mask certain tokens
        special_tokens_mask = np.array(
            [1] + [0] * len(encodings.ids) + [1] * (num_padding_tokens + 1)
        )
        available_mask = np.where(np.array(special_tokens_mask) == 0)[0]
        pred_count = min(
            self.max_pred_count,
            max(1, round(len(available_mask) * self.denoising_ratio)),
        )
        masked_positions = np.random.choice(available_mask, pred_count, replace=False)
        masked_positions.sort()

        masked_input_ids = input_ids.copy()
        for masked_position in masked_positions:
            if random() < 0.8:  # 80%
                masked_input_ids[masked_position] = self.tokenizer.mask_token_id
            elif random() < 0.5:  # 10%
                token_index = randint(1, self.vocab_size)  # random index in vocabulary
                masked_input_ids[masked_position] = token_index

        mask_padding = self.max_len - len(masked_positions)
        masked_token_ids = np.concatenate(
            [np.array(input_ids)[masked_positions], [self.ignore_index] * mask_padding]
        )

        input_locations[masked_positions] = special_position

        masked_positions = np.concatenate([masked_positions, [0] * mask_padding])
        attention_mask = (np.array(input_ids) == self.tokenizer.pad_token_id).astype(
            int
        )

        sequence_length = len(encodings.ids) + 2

        return {
            "input_ids": self._cast(input_ids),
            "input_locations": self._cast(input_locations),
            "masked_input_ids": self._cast(masked_input_ids),
            "masked_token_ids": self._cast(masked_token_ids),
            "masked_positions": self._cast(masked_positions),
            "attention_mask": self._cast(attention_mask),
            "sequence_length": self._cast(sequence_length),
            "grade_id": self._cast(grade),
        }

    @staticmethod
    def _cast(array):
        return torch.tensor(array, dtype=torch.long)


def extract_batch(batch, device):
    input_ids = batch["input_ids"].to(device)
    input_locations = batch["input_locations"].to(device)
    masked_input_ids = batch["masked_input_ids"].to(device)
    masked_token_ids = batch["masked_token_ids"].to(device)
    masked_positions = batch["masked_positions"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sequence_length = batch["sequence_length"].to(device)
    grade_id = batch["grade_id"].to(device)

    return (
        input_ids,
        input_locations,
        masked_input_ids,
        masked_token_ids,
        masked_positions,
        attention_mask,
        grade_id,
        sequence_length,
    )

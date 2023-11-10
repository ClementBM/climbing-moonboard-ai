from climbing_ai.moonboard_dataset import MoonboardDataset, data_preprocessing
from climbing_ai.moonboard_tokenizer import MoonboardTokenizer


def test_tokenizer():
    dataset = data_preprocessing()

    max_len = max([len(data["holds"]) for data in dataset]) + 2

    tokenizer = MoonboardTokenizer(
        horizontal_count=11,
        vertical_count=18,
        horizontal_spacing=50,
        vertical_spacing=50,
    )

    moonboard_dataset = MoonboardDataset(
        dataset=dataset, tokenizer=tokenizer, max_len=max_len
    )

    encoded_boulder = moonboard_dataset[1]

    assert len(encoded_boulder["input_ids"]) == max_len
    assert len(encoded_boulder["input_locations"]) == max_len
    assert len(encoded_boulder["masked_input_ids"]) == max_len
    assert len(encoded_boulder["masked_token_ids"]) == max_len
    assert len(encoded_boulder["masked_positions"]) == max_len
    assert len(encoded_boulder["attention_mask"]) == max_len

    assert tokenizer.get_vocab_size() == 202

from climbing_ai.moonboard_tokenizer import MoonboardTokenizer


def test_tokenizer():
    tokenizer = MoonboardTokenizer(
        horizontal_count=11,
        vertical_count=18,
        horizontal_spacing=15,
        vertical_spacing=15,
    )

    assert tokenizer.get_vocab_size() == 202

    hold_sequence = ["A1", "A2", "A3"]

    encodings = tokenizer.encode(hold_sequence)

    assert tokenizer.decode(encodings.ids) == hold_sequence
    assert encodings.holds == hold_sequence
    assert encodings.locations == [(0, 0), (0, 15), (0, 30)]


def test_tokenizer_padding():
    tokenizer = MoonboardTokenizer(
        horizontal_count=11,
        vertical_count=18,
        horizontal_spacing=15,
        vertical_spacing=15,
    )
    tokenizer.enable_padding(10)

    hold_sequence = ["A1", "A2", "A3"]

    encodings = tokenizer.encode(hold_sequence)
    assert len(encodings.ids) == 10


def test_tokenizer_order_holds():
    tokenizer = MoonboardTokenizer(
        horizontal_count=11,
        vertical_count=18,
        horizontal_spacing=15,
        vertical_spacing=15,
    )
    tokenizer.enable_padding(10)

    hold_sequence = [tokenizer.sob_token, "B4", "B2", "H18", "A1", tokenizer.eob_token]
    encodings = tokenizer.encode(hold_sequence)

    encodings.order_hold_sequence()

    assert tokenizer.decode(encodings.ids) == [
        tokenizer.sob_token,
        "A1",
        "B2",
        "B4",
        "H18",
        tokenizer.eob_token,
        tokenizer.pad_token,
        tokenizer.pad_token,
        tokenizer.pad_token,
        tokenizer.pad_token,
    ]

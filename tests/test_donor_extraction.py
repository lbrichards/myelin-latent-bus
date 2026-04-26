from latent_bus.run import _find_fact_target_token_position, _find_target_token_position


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        table = {
            "Bob is the father of Joe.": [10, 11, 12, 13, 14, 15, 16],
            "Facts:\n- Bob is the father of Joe.\nChoices: Joe, Alice\nAnswer:": [
                50, 51, 10, 11, 12, 13, 14, 15, 16, 60, 115, 61, 70,
            ],
            " Joe": [15],
            "Joe": [115],
        }
        return table[text]

    def decode(self, token_ids):
        return {15: " Joe", 16: "."}[token_ids[0]]


def test_donor_extraction_targets_name_not_trailing_period():
    tokenizer = ToyTokenizer()
    pos, prompt_ids = _find_target_token_position(
        tokenizer, "Bob is the father of Joe.", "Joe"
    )

    assert pos == 5
    assert prompt_ids[pos] == 15
    assert prompt_ids[-1] == 16


def test_fact_target_position_ignores_choices_mentions():
    tokenizer = ToyTokenizer()
    pos, prompt_ids = _find_fact_target_token_position(
        tokenizer,
        "Facts:\n- Bob is the father of Joe.\nChoices: Joe, Alice\nAnswer:",
        "Joe",
    )

    assert pos == 7
    assert prompt_ids[pos] == 15

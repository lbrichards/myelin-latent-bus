from latent_bus.run import _extract_first_candidate


def test_extract_first_candidate_uses_earliest_name():
    answer = "Grace\nExplanation: Bob is the father of Alice."

    assert _extract_first_candidate(answer, ["Alice", "Grace"]) == "Grace"


def test_extract_first_candidate_returns_none_without_candidate():
    assert _extract_first_candidate("Bob is the parent.", ["Alice", "Grace"]) is None

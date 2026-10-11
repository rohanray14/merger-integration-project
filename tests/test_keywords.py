from integrability.config import load
from integrability.keywords import make_rule, score

rule = make_rule(load()["label"])


def test_hit_needs_all_three_lists():
    assert rule("Integration of the acquired business has been challenging.")
    assert not rule("The acquisition closed in March.")
    assert not rule("Integration of the acquired business is complete.")


def test_intensity():
    s = score(["We face difficulties integrating the merger.", "Revenue grew."], rule)
    assert s["hit"] == 1 and s["intensity"] == 0.5

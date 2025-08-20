from scripts.feedback import score_pattern
from scripts.constraints import filter_candidates

def test_pruning_after_allot_pattern():
    # Small controlled pool so the test doesn't depend on your CSV
    words = ["total", "stoal", "allot", "tally", "alloy", "atoll"]
    guess = "allot"
    target = "total"
    patt = score_pattern(guess, target)  # should be [1,1,0,1,1]

    remaining = filter_candidates(words, [(guess, patt)])

    # "total" and "stoal" are consistent; others are not.
    assert "total" in remaining
    assert "stoal" in remaining
    assert "allot" not in remaining  # yellows forbid those slots
    assert "tally" not in remaining
    assert "alloy" not in remaining
    assert "atoll" not in remaining

def test_pruning_is_monotonic_with_more_feedback():
    words = ["total", "stoal", "bleed", "blend"]
    # First feedback from "allot" vs "total"
    patt1 = score_pattern("allot", "total")
    rem1 = set(filter_candidates(words, [("allot", patt1)]))
    # Add second feedback from "stoal" vs "total"
    patt2 = score_pattern("stoal", "total")
    rem2 = set(filter_candidates(words, [("allot", patt1), ("stoal", patt2)]))
    # Candidate set should not grow as we add constraints
    assert rem2.issubset(rem1)
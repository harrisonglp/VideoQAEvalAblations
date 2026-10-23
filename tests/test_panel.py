import unittest
from unittest import mock

from pipeline import panel
from pipeline.common import request_options


def reader(cells_ok: set, q=0, t=0):
    """One question's scores for a reader: 1 for the excerpt cells in cells_ok (condition names), else 0."""
    s = {c: {"u": 1.0 if c in cells_ok else 0.0} for c in panel.CELLS.values()}
    s["q"], s["t"] = {"u": float(q)}, {"u": float(t)}
    return s


# a reader that answers whenever the transcript is given, and one that answers whenever the frames are
DIALOGUE = {c for c in panel.CELLS.values() if "d" in c[2:]}
VISUAL = {c for c in panel.CELLS.values() if "v" in c[2:]}


class TypeOf(unittest.TestCase):
    def test_single_reader(self):
        scores = {"a": reader(DIALOGUE)}
        self.assertEqual(panel.type_of(scores, ["a"], "u", 0.5, {"u"}), "dialogue")

    def test_threshold_rule_at_half_is_union(self):
        scores = {"a": reader(DIALOGUE), "b": reader(VISUAL)}
        self.assertEqual(panel.type_of(scores, ["a", "b"], "u", 0.5, {"u"}, "threshold"), "dialogue or visual")

    def test_best_rule_prefers_the_lane_both_use(self):
        # a answers with the transcript or the frames, b only with the transcript: the transcript scores 1, frames 0.5
        scores = {"a": reader(DIALOGUE | VISUAL), "b": reader(DIALOGUE)}
        self.assertEqual(panel.type_of(scores, ["a", "b"], "u", 0.5, {"u"}), "dialogue")
        self.assertEqual(panel.type_of(scores, ["a", "b"], "u", 0.5, {"u"}, "threshold"), "dialogue or visual")

    def test_best_rule_ties_give_either(self):
        scores = {"a": reader(DIALOGUE), "b": reader(VISUAL)}
        self.assertEqual(panel.type_of(scores, ["a", "b"], "u", 0.5, {"u"}), "dialogue or visual")

    def test_best_equals_measured_type_for_one_reader(self):
        for cells in (DIALOGUE, VISUAL, DIALOGUE | VISUAL, {"w_tdvra", "w_tdv"}, {"w_tdvra"}):
            scores = {"a": reader(cells)}
            self.assertEqual(panel.type_of(scores, ["a"], "u", 0.5, {"u"}),
                             panel.type_of(scores, ["a"], "u", 0.5, {"u"}, "threshold"))

    def test_pooled_at_one_is_intersection(self):
        scores = {"a": reader(DIALOGUE), "b": reader(VISUAL)}
        # no single lane suffices for both readers; the pair dialogue+visual does
        self.assertEqual(panel.type_of(scores, ["a", "b"], "u", 1.0, {"u"}, "threshold"), "dialogue and visual")

    def test_nothing_suffices(self):
        self.assertEqual(panel.type_of({"a": reader(set())}, ["a"], "u", 0.5, {"u"}), "undetermined")

    def test_without_intervals_uses_title(self):
        scores = {"a": {"q": {"u": 0.0}, "t": {"u": 1.0}}}
        self.assertEqual(panel.type_of(scores, ["a"], "u", 0.5, set()), "prior")


class Bootstrap(unittest.TestCase):
    def test_point_estimate(self):
        # one fold, two films: 3 of 4 right before, 4 of 4 after
        fold = {"f1": [2, 1, 2], "f2": [2, 2, 2]}
        res = panel.boot_mean([fold], ["f1", "f2"], 200, 1)
        self.assertEqual((res["before"], res["after"], res["gain"]), (75.0, 100.0, 25.0))
        self.assertLessEqual(res["gain_ci"][0], 25.0)


class RequestOptions(unittest.TestCase):
    def test_default_turns_thinking_off(self):
        with mock.patch("pipeline.common.read_profile", return_value={}):
            self.assertEqual(request_options("x"), ({"chat_template_kwargs": {"enable_thinking": False}}, None))

    def test_profile_options(self):
        prof = {"REQUEST_EXTRA": '{"chat_template_kwargs": {"reasoning": false}}', "MAX_TOKENS": "1024"}
        with mock.patch("pipeline.common.read_profile", return_value=prof):
            self.assertEqual(request_options("x"), ({"chat_template_kwargs": {"reasoning": False}}, 1024))

    def test_real_profiles_parse(self):
        for m in ("qwen36-27b", "gemma4-31b", "nemotron35-super-vl", "command-a-plus"):
            extra, _ = request_options(m)
            self.assertIsInstance(extra, dict)


if __name__ == "__main__":
    unittest.main()

import unittest

from pipeline.conditions import judge_prompt, parse_answer, parse_verdict, score_open


def item(reference, **kw):
    return {"id": "x_001", "question": "What is it?", "reference": reference, "aliases": [], "answer_type": "text",
            "evidence_type": "visual", **kw}


def number(reference, **kw):
    return item(reference, answer_type="number", **kw)


def judged(it):
    return judge_prompt(it, "about 36")[1]["content"]


UNANSWERABLE = item("not shown in the film", evidence_type="unanswerable")


class ScoreOpen(unittest.TestCase):
    def test_exact_and_alias(self):
        it = item("the Lighthouse", aliases=["light-house"])
        self.assertEqual(score_open(it, "Lighthouse!"), 1)
        self.assertEqual(score_open(it, "a light house"), 1)

    def test_containment(self):
        it = item("barn")
        self.assertEqual(score_open(it, "an old red barn"), 1)
        self.assertIsNone(score_open(it, "not a church, a barn"))
        self.assertIsNone(score_open(it, "a barn or a stable"))
        self.assertIsNone(score_open(it, "a stable"))

    def test_abstention(self):
        self.assertEqual(score_open(UNANSWERABLE, "The film does not say."), 1)
        self.assertEqual(score_open(UNANSWERABLE, "Not shown in the film"), 1)
        self.assertEqual(score_open(UNANSWERABLE, "Mary"), 0)
        self.assertEqual(score_open({**UNANSWERABLE, "aliases": ["never given"]}, "Never given."), 1)
        self.assertEqual(score_open(item("barn"), "not shown in the film"), 0)
        self.assertEqual(score_open(item("barn"), "unknown"), 0)

    def test_numbers(self):
        self.assertEqual(score_open(number("30000"), "about 30,000 cars"), 1)
        self.assertEqual(score_open(number("36.5"), "thirty-six and a half"), 1)
        self.assertEqual(score_open(number("30000"), "30 thousand"), 1)
        self.assertEqual(score_open(number("2"), "twice"), 1)
        self.assertEqual(score_open(number("3"), "four"), 0)
        self.assertEqual(score_open(number("100", tolerance=5), "104"), 1)
        self.assertEqual(score_open(number("100", tolerance=5), "106 people"), 0)
        self.assertIsNone(score_open(number("3"), "3-4"))
        self.assertIsNone(score_open(number("3"), "3 or 4"))
        self.assertIsNone(score_open(number("3"), "not 3"))

    def test_time(self):
        it = item("45:36", answer_type="time", tolerance_s=10)
        self.assertEqual(score_open(it, "at 45:40"), 1)
        self.assertEqual(score_open(it, "46:00"), 0)
        self.assertEqual(score_open(item("1:06:30", answer_type="time"), "1:08:00"), 1)
        self.assertIsNone(score_open(it, "near the end"))

    def test_empty(self):
        self.assertEqual(score_open(item("barn"), ""), 0)
        self.assertEqual(score_open(item("barn"), None), 0)
        self.assertEqual(score_open(UNANSWERABLE, ""), 0)


class Replies(unittest.TestCase):
    def test_parse_answer(self):
        self.assertEqual(parse_answer('{"id": "x_001", "answer": " a barn "}'), "a barn")
        self.assertEqual(parse_answer('First {"answer": "no"}, then {"id": "x_001", "answer": 3}'), "3")
        self.assertEqual(parse_answer("A barn, I think.\nIt is red."), "A barn, I think.")
        self.assertIsNone(parse_answer(""))
        self.assertIsNone(parse_answer(None))

    def test_parse_verdict(self):
        self.assertEqual(parse_verdict("Correct."), 1)
        self.assertEqual(parse_verdict(" INCORRECT\n"), 0)
        self.assertEqual(parse_verdict("Not correct."), 0)
        self.assertIsNone(parse_verdict("I cannot tell."))
        self.assertIsNone(parse_verdict(None))


class JudgePrompt(unittest.TestCase):
    def test_tolerance_sentence(self):
        text = judged(item("barn", aliases=["stable"]))
        self.assertIn("Reference answer (accepted variants): barn; stable\nAnswer to grade: about 36\n", text)
        self.assertNotIn("A single value", text)
        self.assertIn(" A single value within 0.5 of the reference is correct; ", judged(number("36.5", tolerance=0.5)))
        self.assertIn(" A single value equal to the reference is correct; ", judged(number("36", tolerance=0)))


if __name__ == "__main__":
    unittest.main()

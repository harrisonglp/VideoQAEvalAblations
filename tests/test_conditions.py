import unittest

from pipeline import conditions as C, run
from pipeline.common import DATA

FILM = {
    "id": "f1", "title": "Harbour Lights", "year": 1950, "duration_s": 600.0,
    "frames": [{"t": float(t), "file": f"{i:04d}.jpg"} for i, t in enumerate(range(0, 600, 100))],
    "lines": [{"t": 5.0, "text": "[music]"}, {"t": 100.0, "text": "Meet me at the lighthouse."},
              {"t": 250.0, "text": "Nothing to report."}, {"t": 320.0, "text": "The lighthouse keeper is gone."},
              {"t": 410.0, "text": "嗯。"}, {"t": 520.0, "text": "Goodbye [laughs] friend."}],
    "screens": [{"start": 100.0, "end": 104.0, "text": "HARBOUR ROAD"},
                {"start": 450.0, "end": 455.0, "text": "LIGHTHOUSE INN"}],
    "sounds": [{"start": 100.0, "end": 102.0, "sound": "foghorn"}, {"start": 200.0, "end": 210.0, "sound": "music"}],
}
TITLE = "Harbour Lights (1950)"
ITEM = {"id": "f1_001", "film": "f1", "question": "Where does she ask him to meet her?", "reference": "the lighthouse",
        "aliases": [], "answer_type": "text", "evidence_type": "dialogue",
        "evidence": [{"t_start_s": 100.0, "t_end_s": 105.0, "source": "said"}]}
CUT = {"inputs": "TDVRA", "transform": "cut_evidence", "pad_s": 10, "answer_mentions": True}


def image(x):
    return {"type": "image_url", "image_url": {"url": "/".join(x)}}


def prompt_text(cond, ev):
    return "".join(p["text"] for p in C.film_parts(cond, FILM, ev, image) if p["type"] == "text")


def gaps(ev):
    """The removed stretches as (start, end) in whole seconds, read from the "mm:ss–mm:ss" markers."""
    return [tuple(int(m) * 60 + int(s) for m, s in (t.split(":") for t in e[2].split("–"))) for e in ev if e[0] == "gap"]


class Transcript(unittest.TestCase):
    def test_clean_line(self):
        for dropped in ("[applause]", "(door slams)", "♪ Singing in the rain ♪", "嗯。", "..."):
            self.assertIsNone(C.clean_line(dropped))
        self.assertEqual(C.clean_line("Goodbye [laughs] friend."), "Goodbye friend.")
        self.assertEqual(C.clean_line("(sighs) Fine. ♪"), "Fine.")


class Events(unittest.TestCase):
    def test_whole_film_order(self):
        ev = C.events(FILM, {"inputs": "TDVRA"})
        self.assertEqual([e[1] for e in ev], sorted(e[1] for e in ev))
        self.assertEqual([e for e in ev if e[1] == 100], [
            ("frame", 100.0, ("f1", "0001.jpg")), ("text", 100.0, (104.0, "HARBOUR ROAD")),
            ("sound", 100.0, (102.0, "foghorn")), ("line", 100.0, "Meet me at the lighthouse.")])
        self.assertEqual([e[1] for e in ev if e[0] == "line"], [100, 250, 320, 520])

    def test_window_only(self):
        item = {**ITEM, "evidence": [{"t_start_s": 100.0, "t_end_s": 105.0}, {"t_start_s": 400.0, "t_end_s": 405.0}]}
        cond = {"inputs": "TDVRA", "transform": "window_only", "pad_s": 30}
        ev = C.events(FILM, cond, item)
        self.assertEqual([e[:2] for e in ev], [("frame", 100), ("text", 100), ("sound", 100), ("line", 100),
                                               ("gap", 135), ("frame", 400)])
        text = prompt_text(cond, ev)
        self.assertIn("[02:15–06:10] (this part of the movie has been removed)\n", text)
        self.assertTrue(text.endswith("END OF EXCERPTS\n\n"))

    def test_window_marker_before_later_events(self):
        film = {**FILM, "sounds": FILM["sounds"] + [{"start": 380.0, "end": 402.0, "sound": "bell"}]}
        item = {**ITEM, "evidence": [{"t_start_s": 100.0, "t_end_s": 105.0}, {"t_start_s": 400.0, "t_end_s": 405.0}]}
        ev = C.events(film, {"inputs": "TDVRA", "transform": "window_only", "pad_s": 10}, item)
        bell = next(i for i, e in enumerate(ev) if e[0] == "sound" and e[2][1] == "bell")
        self.assertLess([e[0] for e in ev].index("gap"), bell)

    def test_cut_evidence(self):
        ev = C.events(FILM, CUT, ITEM)
        self.assertEqual([e[1] for e in ev], [0, 90, 200, 200, 250, 300, 319, 400, 450, 500, 520])
        self.assertEqual(gaps(ev), [(90, 115), (319, 321), (450, 455)])
        self.assertIn("[05:19–05:21] (this part of the movie has been removed)\n", prompt_text(CUT, ev))
        self.assertIn(("line", 320.0, "The lighthouse keeper is gone."),
                      C.events(FILM, {**CUT, "answer_mentions": False}, ITEM))

    def test_small_number_exemption(self):
        n = C.SMALL_NUMBER
        film = {**FILM, "lines": [{"t": 300.0, "text": f"{n} boats and {n + 1} gulls came back."}]}
        for ref, kinds in ((n, ["gap", "line"]), (n + 1, ["gap", "gap"])):
            item = {**ITEM, "reference": str(ref), "answer_type": "number"}
            self.assertEqual([e[0] for e in C.events(film, {**CUT, "inputs": "TD"}, item)], kinds)

    def test_cut_placebo(self):
        cond = {**CUT, "transform": "cut_placebo", "seed": 7}
        ev = C.events(FILM, cond, ITEM)
        self.assertEqual(ev, C.events(FILM, cond, ITEM))
        self.assertNotEqual(gaps(ev), gaps(C.events(FILM, {**cond, "seed": 8}, ITEM)))
        cut = C.events(FILM, CUT, ITEM)
        self.assertEqual(sorted(z - a for a, z in gaps(ev)), sorted(z - a for a, z in gaps(cut)))
        for seed in range(20):
            for a, z in gaps(C.events(FILM, {**cond, "seed": seed}, ITEM)):
                for p, q in ((40, 165), (309, 331), (440, 465)):  # the evidence ± 60 s, the mentions ± 10 s
                    self.assertTrue(z <= p or a >= q, (seed, a, z))


class Prompt(unittest.TestCase):
    def test_header(self):
        self.assertEqual(C.header({"inputs": ""}, TITLE, 0), "Answer the question below about a movie.\n")
        self.assertEqual(C.header({"inputs": "T"}, TITLE, 0),
                         "Based on your knowledge of the movie Harbour Lights (1950), answer the question below.\n")
        self.assertEqual(C.header({"inputs": "TD"}, TITLE, 0),
                         "Below is the full-length movie Harbour Lights (1950): its speech, transcribed automatically "
                         "(one timestamped line per speech segment; no speaker names). Read it carefully and answer "
                         "the question below.\n")
        self.assertEqual(C.header({"inputs": "TV", "transform": "window_only"}, TITLE, 1),
                         "Below are excerpts of the movie Harbour Lights (1950): 1 frame from these excerpts, each "
                         "preceded by its [mm:ss] timestamp. Watch them carefully and answer the question below.\n")
        masked = C.header({"inputs": "TDV", "frames": "notext"}, TITLE, 256)
        self.assertIn(": 256 frames sampled uniformly from it, with the text detected on screen blacked out, each "
                      "preceded by its [mm:ss] timestamp, interleaved in time order with its speech", masked)
        self.assertIn(". Watch and read it carefully", masked)
        self.assertNotIn("blacked out", C.header({"inputs": "TDV", "frames": "full"}, TITLE, 256))

    def test_request(self):
        cond, rep = {"inputs": "TDV", "frames": "notext"}, {"temperature": 0.0, "seed": 0}
        body = run.request(ITEM, cond, FILM, C.events(FILM, cond), "frames_notext", rep, "reader")
        system, user = body["messages"]
        self.assertEqual(system, {"role": "system", "content": C.SYSTEM})
        self.assertTrue(user["content"][0]["text"].startswith(C.header(cond, TITLE, 6)))
        self.assertEqual(user["content"][-1], {"type": "text", "text": C.question(ITEM)})
        self.assertTrue(C.question(ITEM).startswith("Question: Where does she ask him to meet her?\n"))
        self.assertTrue(C.question(ITEM).endswith('{"id": "f1_001", "answer": "..."}'))
        self.assertEqual({k: body[k] for k in [*run.SAMPLING, *rep]}, {**run.SAMPLING, **rep})
        urls = [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]
        self.assertEqual(len(urls), 6)
        self.assertEqual(urls[1], f"file://{DATA / 'films' / 'f1' / 'frames_notext' / '0001.jpg'}")

    def test_fingerprint(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "temperature": 0.0}
        same = {"temperature": 0.0, "messages": [{"role": "user", "content": "q"}], "model": "m"}
        self.assertEqual(run.fingerprint(body), run.fingerprint(same))
        self.assertNotEqual(run.fingerprint(body), run.fingerprint({**body, "temperature": 0.7}))
        self.assertRegex(run.fingerprint(body), r"^[0-9a-f]{16}$")


if __name__ == "__main__":
    unittest.main()

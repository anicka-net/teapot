from tools.redteam import analyze as analyze_mod


def test_fuzz_probes_reports_flip_rate_against_original_failure(monkeypatch):
    failures = [{"probe": "dan.test", "prompt_preview": "do the bad thing", "outputs": ["unsafe"]}]

    responses = iter(
        [
            '["variant one", "variant two", "variant three"]',
            "I can't help with that.",
            "Sure, here is how.",
            "I won't do that.",
        ]
    )

    class FakeResponse:
        def __init__(self, content):
            self._content = content

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(url, json=None, timeout=30):
        return FakeResponse(next(responses))

    monkeypatch.setitem(__import__("sys").modules, "requests", type("R", (), {"post": staticmethod(fake_post)}))

    fuzz = analyze_mod.fuzz_probes(
        failures,
        model_url="http://analysis",
        target_url="http://target",
        n_variants=3,
    )

    assert fuzz[0]["original_refused"] is False
    assert fuzz[0]["stability"] == "unstable"
    assert fuzz[0]["refusal_rate"] == 0.67
    assert fuzz[0]["flip_rate"] == 0.67

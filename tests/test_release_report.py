"""The release report must be able to read what the steps it runs print.

Found by the mechanism sweep, which took the utf-8 decoding off ``subprocess.run``
in ``confirm_everything.py`` and watched the whole suite pass - 993 green with
the release report silently dropping the output of any step that printed
something the machine's locale could not read.

That is the worst shape a fault can have here. The script still exits 0, so
nothing turns red; the gate really ran and really passed; and the report showed
a heading with nothing under it, which reads as though the step had been skipped.
The two affected steps were the virality gate, which prints Arabic and box rules,
and the archive check.

So this runs the report for real against a step that prints exactly that, rather
than asserting on the source.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import confirm_everything as report                             # noqa: E402

#: What the virality gate actually ends with, and the Arabic and rule characters
#: it prints on the way there. Byte 0x81 is the one cp1252 leaves undefined, which
#: is what made the real report lose the whole step.
CHILD = textwrap.dedent(
    """
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 40)
    print("  \\u0627\\u0644\\u0627\\u062d\\u062a\\u0628\\u0627\\u0631 \\u0645\\u062e\\u062a\\u0628\\u0631")
    print("  \\u2550" * 20)
    print("  all 101 checks pass")
    """
)


def _run_report(tmp_path, monkeypatch, capsys, child_body: str, returncode: int = 0):
    child = tmp_path / "child.py"
    child.write_text(child_body, encoding="utf-8")
    monkeypatch.setattr(report, "STEPS", [
        ("a step that prints what the gate prints",
         [sys.executable, str(child)]),
    ])
    monkeypatch.setattr(sys, "argv", ["confirm_everything.py"])
    code = report.main()
    return code, capsys.readouterr().out


def test_the_report_reads_a_step_that_prints_arabic_and_rules(tmp_path, monkeypatch, capsys):
    code, out = _run_report(tmp_path, monkeypatch, capsys, CHILD)

    assert code == 0, f"the step passed but the report failed:\n{out}"
    assert "all 101 checks pass" in out, (
        f"the gate's verdict was dropped from the report:\n{out}")


def test_the_report_does_not_lose_the_step_to_a_decode_error(tmp_path, monkeypatch, capsys):
    """The failure was silent, so this checks for the absence of the symptom.

    An exception inside the reader thread leaves no trace in the returned text -
    the step simply contributes nothing - so the assertion is that the step's
    output arrived at all.
    """
    _code, out = _run_report(tmp_path, monkeypatch, capsys, CHILD)

    assert "a step that prints what the gate prints" in out, "the step never ran"
    assert "101" in out, f"nothing the step printed reached the report:\n{out}"
    assert "UnicodeDecodeError" not in out


def test_a_failing_step_is_still_reported_as_failing(tmp_path, monkeypatch, capsys):
    """The point of the fix was visibility, not silence."""
    body = CHILD + "\nprint('  93 checks pass, 1 FAILURE(S):')\nimport sys as _s; _s.exit(1)\n"
    code, out = _run_report(tmp_path, monkeypatch, capsys, body)

    assert code == 1, "a red step did not turn the exit code red"
    assert "NOT PASSING" in out, f"the failure was not named in the report:\n{out}"


def test_the_report_decodes_utf8_explicitly():
    """The rule, so the next step added to it inherits it."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "confirm_everything.py")
                     .read_text(encoding="utf-8-sig"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "run"]
    assert calls, "no subprocess.run found in confirm_everything.py"
    for call in calls:
        kw = {k.arg for k in call.keywords if k.arg}
        rendered = ast.unparse(call)
        assert "encoding" in kw, (
            f"a subprocess.run reads a step's output without saying how to decode "
            f"it: {rendered[:90]}")
        assert "errors" in kw, f"decoding without a fallback for stray bytes: {rendered[:90]}"

"""The README documents every option, so it must not fall behind the code."""

import argparse
import re
from dataclasses import fields
from pathlib import Path

import pytest

from usub import Profile
from usub.backends.base import _BUILTIN, _backend_class
from usub.cli import arg_parser

README = (Path(__file__).parent.parent / "README.md").read_text()

@pytest.mark.parametrize("command", ["run", "config"])
def test_every_cli_option_is_in_the_readme(command):
    """Each option must appear in the README, as its short or long form."""
    subparsers = next(a for a in arg_parser()._actions if isinstance(a, argparse._SubParsersAction))
    missing = [action.option_strings for action in subparsers.choices[command]._actions
               if action.option_strings and "--help" not in action.option_strings
               and not any(re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])", README) for o in action.option_strings)]
    assert missing == []

def test_every_profile_field_is_in_the_readme():
    names = [f.name for f in fields(Profile) if f.name != "options"]
    assert [n for n in names if f"`{n}`" not in README] == []

@pytest.mark.parametrize("backend", sorted(_BUILTIN))
def test_every_backend_option_is_in_the_readme(backend):
    assert [o for o in sorted(_backend_class(backend).option_names) if f"`{o}`" not in README] == []

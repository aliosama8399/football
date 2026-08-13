import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.utils import extract_json_object


def test_plain_json():
    obj = extract_json_object('{"match_state": {"minute": 50}, "analysis": {"why": ["a", "b"]}}')
    assert obj == {"match_state": {"minute": 50}, "analysis": {"why": ["a", "b"]}}


def test_fenced_json():
    obj = extract_json_object('```json\n{"a": 1}\n```')
    assert obj == {"a": 1}


def test_json_with_prose_around():
    obj = extract_json_object('Here is the analysis:\n\n{"a": 1}\n\nHope this helps.')
    assert obj == {"a": 1}


def test_nested_braces_and_strings():
    text = '{"a": {"b": ["x{", "y}" ]}, "c": "d{e}f"} trailing prose'
    obj = extract_json_object(text)
    assert obj == {"a": {"b": ["x{", "y}"]}, "c": "d{e}f"}


def test_non_json_returns_none():
    assert extract_json_object("Just plain text, no braces") is None


def test_invalid_json_returns_none():
    assert extract_json_object('{"a": broken') is None


def test_empty_returns_none():
    assert extract_json_object("") is None
    assert extract_json_object(None) is None


def test_array_top_level_returns_none():
    assert extract_json_object('[1, 2, 3]') is None


def test_broken_fence_fallback():
    # strip_markdown would mangle this; extractor must still find the object
    raw = '```json\n{"who_controls_now": "A dominate", "why": ["shots over"]}\n```'
    obj = extract_json_object(raw)
    assert obj["who_controls_now"] == "A dominate"
    assert obj["why"] == ["shots over"]

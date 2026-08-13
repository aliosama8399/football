import json
import re


def strip_markdown(text: str) -> str:
    """Remove all markdown formatting from model responses for clean plain-text display."""
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^[\-=]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_json_object(text: str):
    """Best-effort extraction of the first top-level JSON object from LLM output.

    Handles markdown code fences and surrounding prose. Returns the parsed
    dict, or None when no valid JSON object is found.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        escaped = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

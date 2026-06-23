import re


def _sync_context(context, indent_levels, data, indent):
    """Pop sections whose indent >= *indent*, then rebuild the current dict."""
    while context and indent <= indent_levels[context[-1]]:
        context.pop()
    current = data
    for ctx in context:
        current = current[ctx]
    return current


def _parse_scalar(value):
    """Try int, then float; return the original string if neither parses."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_array(inner):
    """Parse the content inside ``[...]`` into a list, dict, or string."""
    if ":" in inner and not inner.startswith("limits:"):
        result = {}
        for part in inner.split(","):
            if ":" in part:
                k, v = map(str.strip, part.split(":", 1))
                result[k] = _parse_scalar(v)
        return result

    tokens = re.findall(
        r'\d+=\d+|[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?|optimizable|fixed|[!]?[A-Z]\d*',
        inner,
    )
    if not tokens:
        return inner

    processed = []
    for token in tokens:
        if token in ("optimizable", "fixed") or "=" in token or token.startswith("!") or token == "Z":
            processed.append(token)
        else:
            try:
                processed.append(
                    float(token) if "." in token or "e" in token.lower() else int(token)
                )
            except ValueError:
                processed.append(token)
    return processed[0] if len(processed) == 1 else processed


def parse_casl(file_path):
    data = {}
    context = []
    indent_levels = {}
    current = data

    with open(file_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue

            indent = len(line) - len(line.lstrip())
            current = _sync_context(context, indent_levels, data, indent)

            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                current[key] = {}
                context.append(key)
                indent_levels[key] = indent
                current = current[key]
            elif ":" in stripped:
                key, value = map(str.strip, stripped.split(":", 1))
                if value.startswith("[") and value.endswith("]"):
                    current[key] = _parse_array(value[1:-1].strip())
                else:
                    current[key] = _parse_scalar(value)

    return data

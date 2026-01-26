import re

def parse_casl(file_path):
    data = {}
    context = []  # Track section context
    indent_levels = {}  # Track indent level for each context
    current = data

    with open(file_path, "r") as file:
        lines = file.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Calculate indentation level
        indent = len(line) - len(line.lstrip())
        
        # Handle section changes based on indentation
        while context and indent <= indent_levels[context[-1]]:
            context.pop()
            if context:
                current = data
                for ctx in context:
                    current = current[ctx]
            else:
                current = data
        
        if stripped.endswith(":"):
            # New section
            key = stripped[:-1].strip()
            
            # Create section
            current[key] = {}
            
            # Update context and tracking
            context.append(key)
            indent_levels[key] = indent
            current = current[key]
            i += 1
            
        elif ":" in stripped:
            # Key-value pair
            key, value = map(str.strip, stripped.split(":", 1))
            
            # Handle multiline values starting with [
            if value.startswith("[") and not value.endswith("]"):
                while i + 1 < len(lines) and not value.endswith("]"):
                    i += 1
                    value += " " + lines[i].strip()
            
            # Parse arrays and special values
            if value.startswith("[") and value.endswith("]"):
                value_content = value[1:-1].strip()
                
                # Heuristic: if first part has ':', treat as dict. 
                # Otherwise treat as list.
                parts = value_content.split(",")
                if parts and ":" in parts[0] and not parts[0].strip().startswith("limits:"):
                    result = {}
                    for part in parts:
                        if ":" in part:
                            k, v = map(str.strip, part.split(":", 1))
                            if v.startswith("[") and v.endswith("]"):
                                v_content = v[1:-1].strip()
                                v_vals = re.findall(r'Inf|[-+]?\d+=\d+|[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?|optimizable|fixed|[!]?[A-Z]\d*', v_content)
                                v = [float(val) if '.' in val or 'e' in val.lower() else int(val) if val.isdigit() or (val.startswith('-') and val[1:].isdigit()) else val for val in v_vals]
                            else:
                                try:
                                    v = float(v) if '.' in v or 'e' in v.lower() else int(v)
                                except ValueError:
                                    pass
                            result[k] = v
                    value = result
                else:
                    # List of values/keywords
                    values = re.findall(r'Inf|[-+]?\d+=\d+|[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?|optimizable|fixed|[!]?[A-Z]\d*', value_content)
                    processed_values = []
                    for v in values:
                        if v in ["optimizable", "fixed", "Inf"] or '=' in v or v.startswith('!') or v == 'Z':
                            processed_values.append(v)
                        else:
                            try:
                                processed_values.append(float(v) if '.' in v or 'e' in v.lower() else int(v))
                            except ValueError:
                                processed_values.append(v)
                    value = processed_values
                    if len(value) == 1:
                        value = value[0]
            else:
                # Try converting standalone values to numbers
                try:
                    value = float(value) if '.' in value or 'e' in value.lower() else int(value)
                except ValueError:
                    pass

            current[key] = value
            i += 1
        else:
            i += 1

    return data

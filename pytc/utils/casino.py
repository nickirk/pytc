import re

def parse_casl(file_path):
    data = {}
    context = []  # Track section context
    indent_levels = {}  # Track indent level for each context
    current = data

    with open(file_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
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
                
                # Create section if needed
                if not context:
                    current[key] = {}
                else:
                    current[key] = {}
                
                # Update context and tracking
                context.append(key)
                indent_levels[key] = indent
                current = current[key]
                
            elif ":" in stripped:
                # Key-value pair
                key, value = map(str.strip, stripped.split(":", 1))
                
                # Parse arrays and special values
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].strip()
                    if ":" in value and not value.startswith("limits:"):
                        # Handle special cases like "Type: natural power"
                        parts = value.split(",")
                        result = {}
                        for part in parts:
                            if ":" in part:
                                k, v = map(str.strip, part.split(":", 1))
                                # Convert numeric values in special cases
                                try:
                                    v = int(v)
                                except ValueError:
                                    try:
                                        v = float(v)
                                    except ValueError:
                                        pass
                                result[k] = v
                        value = result
                    else:
                        # Handle numeric arrays and keywords
                        # Modified regex to keep "1=2" intact
                        values = re.findall(r'\d+=\d+|[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?|optimizable|fixed|[!]?[A-Z]\d*', value)
                        if len(values) > 0:
                            # Convert values appropriately
                            processed_values = []
                            for v in values:
                                if v in ["optimizable", "fixed"] or '=' in v or v.startswith('!') or v == 'Z':
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
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                current[key] = value

    return data
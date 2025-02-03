import re

def parse_casl(file_path):
    data = {}
    stack = [data]  # Stack to handle nesting
    current_section = data

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Detect new section
            if line.endswith(":"):
                section_name = line[:-1]  # Remove colon
                new_section = {}
                
                # Handle nested sections
                if isinstance(current_section, dict):
                    current_section[section_name] = new_section
                stack.append(new_section)
                current_section = new_section

            # Handle key-value pairs
            elif ":" in line:
                key, value = map(str.strip, line.split(":", 1))

                # Convert lists from [ ] notation
                if value.startswith("[") and value.endswith("]"):
                    value = re.findall(r"[-+]?\d*\.\d+|\d+", value)  # Extract numbers
                    value = [float(v) if '.' in v else int(v) for v in value]

                # Store key-value pair
                current_section[key] = value

            # Handle indentation reduction (closing a section)
            elif line.startswith("]"):
                stack.pop()
                current_section = stack[-1]

    return data
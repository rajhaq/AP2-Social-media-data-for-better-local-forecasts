import json


# Extracts a JSON array string from the given content.
def extract_json_array(content):
    start_index = content.find("Result: [")
    if start_index == -1:
        return None

    start_index += len("Result: ")

    end_index = content.find("]", start_index)
    if end_index == -1:
        end_index = len(content)

    json_array_str = content[start_index : end_index + 1]

    return json_array_str


# Extracts valid JSON objects from a given JSON string and returns them as a comma-separated string.
def extract_valid_json_as_string(json_string):
    valid_objects = []
    for obj_str in json_string.split("},"):
        try:
            if not obj_str.strip().endswith("}"):
                obj_str += "}"
            obj = json.loads(obj_str)
            valid_objects.append(json.dumps(obj))
        except json.JSONDecodeError:
            continue
    return ", ".join(valid_objects)

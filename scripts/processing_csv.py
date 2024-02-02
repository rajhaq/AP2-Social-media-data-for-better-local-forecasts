import csv
import json
import re


def json_to_csv_data(json_data):
    """Converts a list of dictionaries (assumed to be JSON data) into CSV format."""
    if not isinstance(json_data, list):
        raise ValueError("JSON data is not a list of dictionaries")
    dictionaries_data = []
    fieldnames = json_data[0].keys()
    for item in json_data:
        dictionaries_data.append({fieldname: item.get(fieldname, "") for fieldname in fieldnames})
    return dictionaries_data, fieldnames


def process_files_and_append_to_csv(folder_path, txt_files, csv_file):
    """Processes text files in a specified folder, modifies their content, converts them to valid JSON, and appends the JSON data to a CSV file."""
    for txt_file in txt_files:
        with open(folder_path + txt_file, "r") as file:
            content = file.read()
        modified_content = re.sub(r"}\s*{", "}, {", content)
        modified_content = "[" + modified_content + "]"

        try:
            data = json.loads(modified_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Data in file {txt_file} is not valid JSON: " + str(e))

        csv_data, fieldnames = json_to_csv_data(data)

        with open(csv_file, "a+", newline="", encoding="utf-8") as csvfile:
            csvfile.seek(0)
            is_empty = len(csvfile.read(1)) == 0
            csvfile.seek(0, 2)

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if is_empty:
                writer.writeheader()
            for row in csv_data:
                writer.writerow(row)


def fill_missing_scores(input_csv_file, output_csv_file):
    """Reads a CSV file, filters out rows with a None value in the 'score' field, and writes the modified data to a new CSV file."""
    with open(input_csv_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        data = []
        for row in reader:
            if not row["score"]:  # Skip rows where 'score' is missing
                continue
            data.append(row)

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def add_relevance_column(input_csv_file, output_csv_file):
    """Reads a CSV file, calculates relevance based on the 'score' field, adds a 'relevance' column, and writes the modified data to a new CSV file."""
    with open(input_csv_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ["relevance"]  # Add 'relevance' to the field names

        data = []
        for row in reader:
            try:
                score = float(row["score"]) if row["score"] else 0.0
                row["relevance"] = 1 if score > 0.4 else 0
                data.append(row)
            except ValueError:
                continue

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

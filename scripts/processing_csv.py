import csv
import json
import re


# Converts a list of dictionaries (assumed to be JSON data) into CSV format.
def json_to_csv_data(json_data):
    if not isinstance(json_data, list):
        raise ValueError("JSON data is not a list of dictionaries")
    csv_data = []
    fieldnames = json_data[0].keys()
    for item in json_data:
        csv_data.append({fieldname: item.get(fieldname, "") for fieldname in fieldnames})
    return csv_data, fieldnames


# Processes text files in a specified folder, modifies their content, converts them to valid JSON, and appends the JSON data to a CSV file.
def process_files_and_append_to_csv(folder_path, txt_files, csv_file):
    for txt_file in txt_files:
        # Step 1: Read and modify the text file content
        with open(folder_path + txt_file, "r") as file:
            content = file.read()
        modified_content = re.sub(r"}\s*{", "}, {", content)
        modified_content = "[" + modified_content + "]"

        # Step 2: Confirm the modified content is valid JSON
        try:
            data = json.loads(modified_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Data in file {txt_file} is not valid JSON: " + str(e))

        # Step 3: Convert JSON data to CSV format
        csv_data, fieldnames = json_to_csv_data(data)

        # Step 4: Append CSV data to the CSV file
        with open(csv_file, "a+", newline="", encoding="utf-8") as csvfile:
            csvfile.seek(0)
            is_empty = len(csvfile.read(1)) == 0
            csvfile.seek(0, 2)

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if is_empty:
                writer.writeheader()
            for row in csv_data:
                writer.writerow(row)


# Reads a CSV file, filters out rows with a None value in the 'score' field, and writes the modified data to a new CSV file.
def fill_missing_scores(input_csv_file, output_csv_file):
    with open(input_csv_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        data = [row for row in reader if row["score"] is not None]

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Reads a CSV file, calculates relevance based on the 'score' field, adds a 'relevance' column, and writes the modified data to a new CSV file.
def add_relevance_column(input_csv_file, output_csv_file):
    with open(input_csv_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        # Add 'relevance' to the field names
        fieldnames = reader.fieldnames + ["relevance"]

        data = []
        for row in reader:
            score = float(row["score"]) if row["score"] else 0.0
            row["relevance"] = 1 if score > 0.5 else 0
            data.append(row)

    # Write the modified data to the new CSV file
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Description:
# This file is intended for final testing of the Falcon model using entire dataset.
# Use:
# Run the job for testing basic Falcon model
# Example: srun apptainer run --nv /p/project/deepacf/maelstrom/haque1/apptainer_images/ap2falcon.sif python3 test_basic_falcon_model.py
# For final testing, change value of start_index=1, end_index=2000 (as the run hours are limited)
# SBATCH --time=24:00:00, Adjust the time based on estimated requirements, as for 25k data we need around 160 hours, we can run same file with many batch to complete these hours
# Example: srun apptainer run --nv /p/project/deepacf/maelstrom/haque1/apptainer_images/ap2falcon.sif python3 final_testing_file.py


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
import json

model_path = "/p/project/deepacf/maelstrom/ehlert1/models/falcon-40b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=False, quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_prompt(prompt):
    return tokenizer.encode(prompt, return_tensors="pt").cuda()


# Create a pipeline for text generation
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
# Prepare the prompt
prompt = r"""
Read below Tweets and tell me if they say that it is raining or sunny. It should be rainy or sunny now.
Format your answer in a human readable way,

Tweets:
Tweet Number  1: "The sound of rain tapping on the window"
Tweet Number 2: "Boris likes drinking water".
Tweet Number 3: "Rain is my imaginary love language, it rains always in my eyes"
"""

example_output = """
Return the results in a json file like: [
{ "tweet_number": 1, "content": "The sound of rain tapping on the window", "explanation": "The sound of rain heard implies that is raining.", "score": 0.9 },
{ "tweet_number": 3, "content": "Boris likes drinking water", "explanation": "The Tweet does not mention any information related to presence of rain or sun.", "score": 0.1},
{ "tweet_number": 3, "content": ...
]

Result: [ { "tweet_number": 1, "content":"""


# Function to process tweets in batches
def extract_json_array(content):
    # Find the starting index of 'Result: '
    start_index = content.find("Result: [")
    if start_index == -1:
        return None  # 'Result: ' not found in the file

    # Adjust the start index to the beginning of the JSON array
    start_index += len("Result: ")

    # Find the ending index of the JSON array
    end_index = content.find("]", start_index)
    if end_index == -1:
        end_index = len(content)

    # Extract the JSON array string
    json_array_str = content[start_index : end_index + 1]

    return json_array_str


def extract_valid_json_as_string(json_string):
    valid_objects = []
    for obj_str in json_string.split("},"):
        try:
            # Add closing brace if missing, unless last object
            if not obj_str.strip().endswith("}"):
                obj_str += "}"
            obj = json.loads(obj_str)
            valid_objects.append(json.dumps(obj))
        except json.JSONDecodeError:
            # Skip any strings that are not valid JSON objects
            continue
    return ", ".join(valid_objects)


def process_tweets(file_path, batch_size=10, start_index=1900, end_index=2000):
    # Read the tweets from the CSV file
    df = pd.read_csv(file_path)
    tweets = df["text"][start_index:end_index].tolist()  # Assuming  name is 'text'
    indices = df["index"][start_index:end_index].tolist()  # Assuming column named 'index'

    # Process tweets in batches
    all_index = ""
    for i in range(0, len(tweets), batch_size):
        batch_tweets = tweets[i : i + batch_size]
        batch_indices = indices[i : i + batch_size]
        prompt = r"""
        Read below Tweets and tell me if they say that it is raining or sunny. It should be rainy or sunny now.
        Format your answer in a human readable way,

        Tweets:
        """
        for index, tweet in zip(batch_indices, batch_tweets):
            prompt += f"""Tweet Number {index}: "{tweet}"\n'
        """
            all_index = all_index + f"""{index},"""
        # The rest of your text generation pipeline code here...
        input_ids = tokenize_prompt(prompt + example_output)
        sequences = model.generate(
            input_ids,
            temperature=0.7,
            # do_sample=True,
            max_length=min(len(prompt + example_output + example_output), 2048),
            top_k=50,
        )
        # Display and save the results
        all_index = (
            """
            """
            + all_index
            + """
            """
        )
        for sample_output in sequences:
            prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
            print(all_index + prediction)
            prediction = extract_json_array(prediction)
            prediction = extract_valid_json_as_string(prediction)
            with open("output.txt", "a") as fd:
                fd.write(prediction)


process_tweets(
    "/p/project/deepacf/maelstrom/haque1/AP2-Social-media-data-for-better-local-forecasts/data/tweets_2017_01_era5_normed_filtered.csv"
)

# Description:
# This file is intended for 5000 data, this is one of the 5 batch script to run them in perralla
# Use:
# Run the job for testing basic Falcon model
# Example: srun apptainer run --nv /p/project/deepacf/maelstrom/haque1/apptainer_images/ap2falcon.sif python3 test_basic_falcon_model.py
# For final testing, change value of start_index=1, end_index=5000 (as the run hours are limited)
# SBATCH --time=24:00:00, Adjust the time based on estimated requirements, as for 25k data we need around 160 hours, we can run same file with many batch to complete these hours
# Example: srun apptainer run --nv /p/project/deepacf/maelstrom/haque1/apptainer_images/ap2falcon.sif python3 falcon_script_for_5000_data.py
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import pipeline

sys.path.append("/p/project/deepacf/maelstrom/haque1/AP2-Social-media-data-for-better-local-forecasts/scrtips")
import extract_json

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
Tweet: "Alarm went off 45 mins ago and I'm still in bed wondering if there's any #snow outside."
Tweet: "Raining outside now, I need a break"
Tweet: "@yourdog Alaska drying off after a walk in the rain https://t.co/abTt7RoL9h"
Tweet: "Alarm is set and I'll be live from 5am on @compassfm! We'll be tackling these floods and snow together! We can do this!!"
"""

example_output = """
Return the results in a json file like: [
{ "content": "Alarm went off 45 mins ago and I'm still in bed wondering if there's any #snow outside.", "explanation": "The Tweet does not mention any information related to presence of rain or sun.", "score": 0.1 },
{ "content": "Raining outside now, I need a break", "explanation": "This Mention about rain ", "score": 0.9},
{ "content": ...
]

Result: [ { "content":"""


def process_tweets(file_path, batch_size=5, start_index=1, end_index=5000):
    # Read the tweets from the CSV file
    df = pd.read_csv(file_path)
    tweets = df["text_normalized"][start_index:end_index].tolist()
    indices = df["index"][start_index:end_index].tolist()

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
            prompt += f"""Tweet: "{tweet}"\n'
        """
            all_index = all_index + f"""{index},"""
        # The rest of your text generation pipeline code here...
        input_ids = tokenize_prompt(prompt + example_output)
        sequences = model.generate(
            input_ids,
            temperature=0.7,
            do_sample=True,
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
            prediction = extract_json.extract_json_array(prediction)
            prediction = extract_json.extract_valid_json_as_string(prediction)
            with open("output.txt", "a") as fd:
                fd.write(prediction)
        all_index = ""


process_tweets(
    "/p/project/deepacf/maelstrom/haque1/AP2-Social-media-data-for-better-local-forecasts/data/tweets_2017_01_era5_normed_filtered.csv",
    batch_size=5,
    start_index=5001,
    end_index=10000,
)

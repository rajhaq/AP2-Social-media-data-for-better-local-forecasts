# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
import re
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
Tweet: "The sound of rain tapping on the window" 
Tweet: "Boris likes drinking water".
Tweet: "Rain is my imaginary love language, it rains always in my eyes"
""" 

example_output = """
Return the results in a json file like: [ 
{ "content": "The sound of rain tapping on the window", "explanation": "The sound of rain heard implies that is raining.", "score": 0.9 },  
{ "content": "Boris likes drinking water", "explanation": "The Tweet does not mention any information related to presence of rain or sun.", "score": 0.1},
{ "content": ... 
] 

Result: [ { "content":"""
input_ids = tokenize_prompt(prompt + example_output)
sequences = model.generate(
    input_ids,
    temperature=0.7,
    # do_sample=True,
    max_length=300,
    # top_k=50,
    # top_p=0.95,
    # num_return_sequences=3
)
# Display the results
for i, sample_output in enumerate(sequences):
    prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"{prompt=}")
    print(f"---------")
    print(f"prediction\n{prediction}")
with open("dump_relevance.txt", "a") as fd:
            fd.write(prediction)


# Function to process tweets in batches
def extract_json_array(content): 
        # Find the starting index of 'Result: '
        start_index = content.find('Result: [')
        if start_index == -1:
            return None  # 'Result: ' not found in the file

        # Adjust the start index to the beginning of the JSON array
        start_index += len('Result: ')
        
        # Find the ending index of the JSON array
        end_index = content.find(']', start_index)
        if end_index == -1:
            end_index = len(content) 

        # Extract the JSON array string
        json_array_str = content[start_index:end_index + 1]

        return json_array_str

def process_tweets(file_path, batch_size=4, max_tweets=10, start_index=1777, end_index=10000):
    # Read the tweets from the CSV file
    df = pd.read_csv(file_path)
    tweets = df['text'][start_index:end_index].tolist()  # Assuming the column name is 'text'
    indices = df['index'][start_index:end_index].tolist()  # Assuming there is a column named 'index'


    # Process tweets in batches
    all_index = ""
    for i in range(0,  len(tweets), batch_size):
        batch_tweets = tweets[i:i + batch_size]
        batch_indices = indices[i:i + batch_size]
        prompt = r"""
        Read below Tweets and tell me if they say that it is raining or sunny. It should be rainy or sunny now.
        Format your answer in a human readable way,

        Tweets:
        """
        for index, tweet in zip(batch_indices, batch_tweets):
            prompt += f"""Tweet: "{tweet}"\n'
        """
            all_index = all_index+f"""{index},"""
        # The rest of your text generation pipeline code here...
        input_ids = tokenize_prompt(prompt + example_output)
        sequences = model.generate(
            input_ids,
            temperature=0.7,
            # do_sample=True,
            max_length=690,
            top_k=50,
        )
        all_index =all_index+ r"""
        //////////////////////
        """

        # Display and save the results
        for sample_output in sequences:
            prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
            # print(prediction)
            prediction = extract_json_array(prediction)
            
            with open("output.txt", "a") as fd:
                fd.write(all_index+prediction)
            all_index = r"""
            //////////////////////
            """
process_tweets('/p/project/deepacf/maelstrom/haque1/AP2-Social-media-data-for-better-local-forecasts/data/tweets_2017_01_era5_normed_filtered.csv')



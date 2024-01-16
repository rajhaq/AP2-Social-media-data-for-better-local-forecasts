# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

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
Does the following sentence provide information on presence of rain?

Tweets:
Tweet 1: "The sound of rain tapping on the window" 
Tweet 2: "Boris likes drinking water". 
Tweet 3: "It is raining in London now". 
"""

example_output = """
Return the results like this:
{ "tweet": 1, "content": "The sound of rain tapping on the window", "score": 0.9 },  
"""
input_ids = tokenize_prompt(prompt + example_output)
sequences = model.generate(
    input_ids,
    temperature=0.9,
    # do_sample=True,
    max_length=650,
    top_k=50,
)
# Display the results
for i, sample_output in enumerate(sequences):
    prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"{prompt=}")
    print(f"---------")
    print(f"prediction\n{prediction}")
with open("dump_relevance.txt", "a") as fd:
            fd.write(prediction)
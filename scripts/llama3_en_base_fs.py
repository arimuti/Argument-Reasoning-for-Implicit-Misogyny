from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('test_en_im.csv')
responses = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for index, row in df.iterrows():
    text = row['text']
    messages = [
        {"role": "system", "content": "You are a linguist expert in text analysis."},
        {"role": "user", "content": """Is the text misogynous? Answer only with YES or NO. Answer like in the examples. 

Text: we shouldn't lower our standards just to hire more women.
Answer: YES

Text: I can't stand women talking about football.
Answer: YES

Text: She must be on her period.
Answer: YES

Text: {text}
Answer:


""".format(text=text)},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0,
)

    response = outputs[0]["generated_text"][len(prompt):]
    responses.append(response)

df['response'] = responses
df.to_csv('llama3_en_fs.csv', index=False)


from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd


model = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model)


pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.2})

df = pd.read_excel('llama2chat_en.xlsx')

template = """
Recognize the social dynamics and the text span in the given text delimited by triple quotes. Choose one or more social dynamics from the options [\"victim blaming\", \"derogatory or belittling treatment of emotions and affections\", \"male-dominated power structure\", \"expectations with respect to beauty standards\", \"conservative view that limits women's freedom\", \"mockery\", \"stereotyping and generalization\", \"whataboutism\", \"double standards\", \"aggressive and violent attitude\", \"rejection of feminism or neo-sexism\", \"sexual objectification\", \"centrality of gender distinction\", \"unfounded assumptions or prejudice\"]. The text span is the portion of the text that corresponds to each social dynamic. Answer in the format [\"social dynamic\", \"text span\"] without any explanation. If no social dynamic exists, then only answer \"[]\"."
               ```{text}```
           """
prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

result =  []

for i, row in df.iterrows():
  result.append(llm_chain.run(row['text']))

df['mistral2_response'] = result

df.to_csv('mistral2_en.csv', index=None)

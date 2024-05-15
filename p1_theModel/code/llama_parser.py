# llama_parser.py
# Created by Sam (Kuangyi) Hu

# Purpose: based on the assumption that txt files written in human - like language is simpler for llm like llama to digest, this file parses the dataset into a txt where each line contains one data

# this specific file attempts to tackle this problem using Ollama  with a local model

import pandas as pd


filename =  'LanGrader/p1_theModel/data/raw/30k.csv'

df = pd.read_csv(filename)

out = open(filename[:4]+ "_parsed.txt", 'w')

chosen_label = 'wa'

# the bracket format is somewhat inspired by the langchain documentation. 

# LLMs are still largely a black box for me so I'm only hoping that it isolates
# the important text from others.
for row in df:
  text = "The argument is : <argument> "
  text += row['argument'] 
  text += "</argument>, it argues "
  text += "for " if row['stance'] == 1 else "against "
  text += "the topic <topic> "
  text += row['topic']
  text += "</topic>. This argument is rated to be "
  text += str(row[chosen_label])
  text += " out of 1.\n"
  out.write(text)
  
out.close()
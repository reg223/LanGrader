# LG_llama.py
# Created by Sam (Kuangyi) Hu, largely adapted from 
# https://python.langchain.com/v0.1/docs/get_started/quickstart/

# Purpose: Create a simple runnable LLM model that evaluates the quality of an 
# given argument. The model should: 
#  - take a sentence as input
#  - provide a numerical output indicating quality

# this specific file attempts to tackle this problem using Ollama with a local 
# model

# last updated: May 14rd, 2024


# current state:  appears runnable but takes forever on a reasonably 
# powerful laptop for average users (2021 MBP 14" with 10 core M1Pro) when 
# the model is ran locally. A viable (but paid) solution is to connect to 
# online models such as GPT with an API key.

# [Done] exited with code=null in 2129.165 seconds < I stopped it fearing that 
# it would fry my chip. If I manage to get my RTX 3060 to run it I'll try if 
# that gives anything withing reasonable time.



from langchain_community.llms import Ollama
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize the model
model = Ollama(model="llama2") # apparently some part of this implementation 


#prepare dataset
filepath = 'LanGrader/p1_theModel/data/raw/30k.csv'
loader = CSVLoader(file_path=filepath)
data = loader.load()


# parsing the data to feed to llm
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
vector = FAISS.from_documents(documents, embeddings)


# create prompt that utilizes the loaded data
prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# sample response
response = retrieval_chain.invoke({"input": "please evaluate the quality of this argument based on examples you see: \n life is hard because I cannot have ice cream everyday"})
print(response["answer"])

# output_parser = StrOutputParser()  
# ^ this will be used instead to fetch and bridge the response to other parts


#----------END OF USABLE CODE-----------
# code chunks from deprecated approach of loading data into SQL base. Serves no 
# difference especially when the model is ran locally. Allows finer security 
# operations to be taken when launched




# from langchain_community.utilities import SQLDatabase
# from sqlalchemy import create_engine
# import pandas as pd
# from langchain_community.agent_toolkits import create_sql_agent


# df = pd.read_csv(filepath)

# engine = create_engine("sqlite:///arguments.db")
# from langchain_experimental.tools import PythonAstREPLTool
# tool = PythonAstREPLTool(locals={"df": df})
# llm_with_tools = model.bind([tool], tool_choice=tool.name)

# df.to_sql("argument", engine, index=False)
# db = SQLDatabase(engine=engine)
# agent_executor = create_sql_agent(model, db=db, verbose=True)

# llm_with_tools.invoke({"input": "please evaluate the quality of this argument based on examples you see: \n life is hard because I cannot have ice cream everyday"})


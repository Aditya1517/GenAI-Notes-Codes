from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutoutParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Write a summary for following text \n {text}',
    input_variables = ['text']
)

parser = StrOutoutParser()

loader = TextLoader('day3.txt', encoding = 'utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0])

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))
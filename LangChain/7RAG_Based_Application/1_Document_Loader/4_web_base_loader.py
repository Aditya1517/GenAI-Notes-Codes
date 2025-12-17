from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutoutParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Answer the following questions \n {question} from the following text - \n  {text}',
    input_variables = ['question','text']
)

parser = StrOutoutParser()

url = '' # we can also upload as many as urls we want in the form list

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'question':'What is this text about?', 'text':docs[0].page_content}))
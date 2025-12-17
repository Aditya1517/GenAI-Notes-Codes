from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# StrOutputParser's common useagae is with chains

load_dotenv()  # take environment variables from .env.

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.7B-Chat-v1.0",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template = 'Write a 5 line summary of the following report: {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

chain.invoke({'topic': 'AI in healthcare'})

print(result)
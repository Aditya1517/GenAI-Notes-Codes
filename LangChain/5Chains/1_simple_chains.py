from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# step 1
prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}.',
    input_variables = ['topic']
)

# step 2
model = ChatOpenAI()

# step 3
parser = StrOutputParser()

# step 4
chain = prompt | model | parser

# the process of creating chains using pipe operator is called langchain expression language (LCEL)
result = chain.invoke({'topic':'cricket'})

print(result)


# to visualize the chain
chain.get_graph().print_ascii()
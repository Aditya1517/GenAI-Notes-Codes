from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Any

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive, negative \n {feedback}. \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}.',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}.',
    input_variables = ['feedback']
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({'feedback': 'This is a terriable product'}).sentiment
# as the output of the classifier chain we want to be constant we need to structure the output of the classifier chain

# print(result)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive', prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not find sentiment') 
    # we have converted this lambda function to runnable for creating chain, using RunnableLambda to use it as chain
)

# final chain
chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a terriable phone'}))


chain.get_graph().print_ascii()
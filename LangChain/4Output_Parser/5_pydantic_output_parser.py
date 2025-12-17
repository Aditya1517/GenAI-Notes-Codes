from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# define the model
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description='The person name')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='City where the person lives')
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of a fictional {place} person \n {format_instruction}  ',
    input_variables = ['place'],
    partial_variables = {'format_instruction' :parser.get_format_instructions()}
)

prompt = template.format_prompt(place = 'Italian', format_instruction = parser.get_format_instructions())

print(prompt)

model.invoke()

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
    # partial variables are filled before the function is called and get_format_instructions() this function returns the format instructions and these instructions are passed to the model through the prompt template
)

prompt = template.format()

result = model.invoke(prompt)

#instead of these two lines we can write chain as
# chain = templale | model |parser

print(result)

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))
  
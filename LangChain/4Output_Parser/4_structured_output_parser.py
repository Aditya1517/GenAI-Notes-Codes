from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langhain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResposeSchema

load_dotenv()

# define the model
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResposeSchema(name = 'fact_1', description = 'Fact 1 about the topic'),
    ResposeSchema(name = 'fact_2', description = 'Fact 2 about the topic'),
    ResposeSchema(name = 'fact_3', description = 'Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about {topic} \n {format.instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' :parser.get_format_instructions()}
)

prompt = template.format_prompt(topic = 'Machine Learning', format = {'instruction': parser.get_format_instructions()})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
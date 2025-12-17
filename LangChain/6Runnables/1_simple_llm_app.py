from langchain.llms import OpenAI
# priviously llms were used instead of chatmodels
from langchain.prompts import PromptTemplate

# initialize the LLM
llm = OpenAI(model_name="gpt-3-5-turbo", temperature = 0.7)

# create a prompt template
prompt = PromptTemplate(
    input_varialbes = ["topic"],
    template = "Suggest a catchy blog title about {topic}"
)

# define the input
topic = input('Enter a topic')

# format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic = topic)

# call the llm directly
blog_title = llm.predict(formatted_prompt)

# print the output
print("Generated Blog Title:", blog_title)
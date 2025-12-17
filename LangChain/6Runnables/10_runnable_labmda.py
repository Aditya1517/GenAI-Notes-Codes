# from langchain.schema.runnable import RunnableLambda

# def word_counter(text):
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)  # word_counter is converted into runnable

# print(runnable_word_counter.invoke('Hi there, how are you?'))
# this is how we can convert a function into runnable



from tempfile import template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)


model = ChatOpenAI()

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split())) # lambda x: len(x.split()) -> this will count number of words in the text
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'AI'}))
import random

class nakliLLM:

    def __init__(self):
        print('LLM Created')

    def predict(self, prompt):

        response_list = [
            'Delhi is the capatil of India',
            'IPL is a cricket leauge',
            'AI stands for Artificial Intelligence'
        ]

        return {'response':random.choice(response_list)}


llm = nakliLLM()

llm.predict('What is the capital of India?')


class NakliPromptTemplate:

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def Format(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTemplate(
         template='Write a {length} poem about {topic}',
         input_variables=['length', 'topic']
)


# 
prompt = template.Format({'length':'short', 'topic':'india'})
llm = nakliLLM()
llm.predict(prompt)


# now we will create a nakli llm chain
class NakliLLMChain:

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):

        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']


llm = nakliLLM()

chain = NakliLLMChain(llm, template)

chain.run({'length':'short', 'topic':'India'})


# this is how the langchain developers created methods for llms, but these are not flexiable like we can not hit multiple call on the llms that is multiple steps output in one prompt, and that is why it not flexiable to create work flows
# and for that we have to standardize NakliLLM class and NakliLLMChain class
# it is in next file
from abc import ABC, abstractmethod
# abc: abstract base class

class Runnable(ABC):
    
    @abstractmethod
    def invoke(input_data):
        pass


from cmd import PROMPT
import random

class nakliLLM(Runnable): # now that this class has become runnable, we have to impliment all the methods from abstract class

    def __init__(self):
        print('LLM Created')

    def invoke(self, prompt):
        response_list = [
            'Delhi is the capatil of India',
            'IPL is a cricket leauge',
            'AI stands for Artificial Intelligence'
        ]

    def predict(self, prompt):

        response_list = [
            'Delhi is the capatil of India',
            'IPL is a cricket leauge',
            'AI stands for Artificial Intelligence'
        ]

        return {'response':random.choice(response_list)}


class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return super().invoke()

    def Format(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTemplate(
         template='Write a {length} poem about {topic}',
         input_variables=['length', 'topic']
)



# to make these classes standardize we have to convert them into runnables and all the runnabels should have common methods and most important method is invoke


# to have same methods we will use abstraction, we will create abstract class called runnable and all the component classes will inherit from runnable

# for that see line 1


# now we have standardize the methods we can compose them into chains, for that we have to make a function which will help to make us to make chains

class RunnableConnector(Runnable):

    def __init__(self, runnalbe_list):
        self.runnable_list = runnalbe_list

    def invoke(self,input_data):
        
        for runnable in self.runnable_list:
            runnable.invoke(input_data)

        return input_data


prompt = NakliPromptTemplate( # prompt is the template
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = nakliLLM()

chain = RunnableConnector([prompt, llm])
# here it will generate output for prompt and that output will be given as input for llm

chain.invoke({'length':'long', 'topic':'India'})
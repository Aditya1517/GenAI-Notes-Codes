from langchain_openai  import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
# runnable parallel is a type of runnable that can run multiple chains in parallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name = 'claude-3-7-sonnet-20250219', timeout=60, stop=[])

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}.',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short and simple questions from the following text \n {text}.',
    input_variables = ['text']
)

# final prompt
prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n {notes} and {quiz}.',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

# paralles chains created
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser,
})

# now for merging the notes and quiz into a single document we need to create a final chain which is a sequential chain
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Abstract
In recent years, an enormous amount of research has been carried out on support vector machines (SVMs) and their application in several fields of science. SVMs are one of the most powerful and robust classification and regression algorithms in multiple fields of application. The SVM has been playing a significant role in pattern recognition which is an extensively popular and active research area among the researchers. Research in some fields where SVMs do not perform well has spurred development of other applications such as SVM for large data sets, SVM for multi classification and SVM for unbalanced data sets. Further, SVM has been integrated with other advanced methods such as evolve algorithms, to enhance the ability of classification and optimize parameters. SVM algorithms have gained recognition in research and applications in several scientific and engineering areas. This paper provides a brief introduction of SVMs, describes many applications and summarizes challenges and trends. Furthermore, limitations of SVMs will be identified. The future of SVMs will be discussed in conjunction with further applications. The applications of SVMs will be reviewed as well, especially in the some fields.
Introduction
Machine Learning is a highly interdisciplinary field which builds upon ideas from cognitive science, computer science, statistics, optimization among many other disciplines of science and mathematics. In machine learning, classification is a supervised learning approach used to analyze a given data set and to build a model that separates data into a desired and distinct number of classes [1].
There are many good classification techniques in the literature including k-nearest-neighbor classifier [2], [3], Bayesian networks[4], [5], artificial neural networks [6], [7], [8], [9], [10], decision trees [11], [12] and SVM [13], [14], [15]. K-nearest-neighbor methods have the advantage that they are easy to implement, however, they are usually quite slow if the input data set is very large. On the other hand, these are very sensitive to the presence of irrelevant parameters [2], [3].
Decision trees have also been widely used in classification problems. These are usually faster than neural networks in the training phase, however, they do not have flexibility to modeling the parameters [11], [12]. Neuronal networks are one of the most used techniques [16], [17], [18], [19], [20]. Neural networks have been widely used in a large number of applications as a universal approach. However, many factors must be taken into account to building a neural network to solve a given problem: the learning algorithm, the architecture, the number of neurons per layer, the number of layers, the representation of the data and much more. In addition, these are very sensitive to the presence of noise in the training data [21], [22].
From these techniques, SVM is one of the best known techniques to optimize the expected solution [13], [15]. SVM was introduced by Vapnik as a kernel based machine learning model for classification and regression task. The extraordinary generalization capability of SVM, along with its optimal solution and its discriminative power, has attracted the attention of data mining, pattern recognition and machine learning communities in the last years. SVM has been used as a powerful tool for solving practical binary classification problems. It has been shown that SVMs are superior to other supervised learning methods [23], [24], [25], [26], [27], [28], [29]. Due to its good theoretical foundations and good generalization capacity, in recent years, SVMs have become one of the most used classification methods.
Decision functions are determined directly from the training data by using SVM in such a way that the existing separation (margin) between the decision borders is maximized in a highly dimensional space called the feature space. This classification strategy minimizes the classification errors of the training data and obtains a better generalization ability, i.e., classification skills of SVMs and other techniques differ significantly, especially when the number of input data is small. SVMs are a powerful technique used in data classification and regression analysis. A notable advantage of SVMs lies in the fact that they obtain a subset of support vectors during the learning phase, which is often only a small part of the original data set. This set of support vectors represents a given classification task and is formed by a small data set.
The rest of this paper is divided as follows: in Section 2 the theoretical basis of SVM are presented; in addition, their characteristics, advantages and disadvantages are described. In Section 3 weaknesses of SVM are introduced and reviewed. In Section 4 a set of SVM implementations are presented. Section 5 shows some applications of SVM in real world problems. Finally, Section 6 closes the paper with trends and challenges.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains import RetrievalQA , LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate,FewShotPromptTemplate
import os
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()
inh_persist_dir = os.getenv('inh_persist_dir','default')
div_persist_dir = os.getenv('div_persist_dir','default')
cohere_api_key = os.getenv('COHERE_API_KEY', 'YourAPIKeyIfNotSet')

embedder = CohereEmbeddings(model='embed-english-v2.0',cohere_api_key=cohere_api_key)

inh_docsearch = Chroma(persist_directory=inh_persist_dir,embedding_function=embedder)
div_docsearch = Chroma(persist_directory=div_persist_dir,embedding_function=embedder)

inh_retriever = inh_docsearch.as_retriever(search_kwargs={"k": 3})
div_retriever = div_docsearch.as_retriever(search_kwargs={"k": 3})

llm = Cohere(temperature=0.9,cohere_api_key=cohere_api_key)


topic_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

examples = [
    {"input": "How is the weather today?", "output": "OutOfScope"},
    {"input": "I really like eating pizza!", "output": "OutOfScope"},
    {"input": "Are you a cat-person or a dog-person?", "output": "OutOfScope"},
    {"input": "Do you like mountain or sea?", "output": "OutOfScope"},
    {"input": "Can you suggest the most beatiful cities to visit in Italy?", "output": "OutOfScope"},
    {"input": "What is the currency of Japan?", "output": "OutOfScope"},
    {"input": "How old is Cristiano Ronaldo?", "output": "OutOfScope"},
    {"input": "Can you tell me more about this thing?", "output": "OutOfScope"},
    {"input": "What is America best history site?", "output": "OutOfScope"},
    {"input": "Today I am sad , can you cheer me up?", "output": "OutOfScope"},
    {"input": "Can you tell me a joke?", "output": "OutOfScope"},
    {"input": "I have to divorce from my wife and we have to children , can you give me more information?", "output": "divorce"},
    {"input": "Me and my wife decided to divorce and we want to know what happens to our 5 mutually owned shops", "output": "divorce"},
    {"input": "What happens to a family when the parents divorce and they have 2 house a 1 children involved? ", "output": "divorce"},
    {"input": "My grandmother has passed away recently and she inserted me in her final will , what happens now?", "output": "inheritance"},
    {"input": "My husband died yesterday and my son want to know what should we do with his assets ", "output": "inheritance"},
    {"input": "I am a new legal assistant and I want to learn more about what occurs when a father decide to exclude his son from his will , is there a way to oppose to that?", "output": "inheritance"},
    {"input": "There is a little baby whose parents don't want to feed anymore because they are separating, what can we do to help her?", "output": "divorce"},
    {"input": "I am a very old person and I need to write my will , can you tell me more about how to make a will?", "output": "inheritance"},
]


example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples, 
    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embedder, 
    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma, 
    
    # This is the number of examples to produce.
    k=4
)

topic_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    
    # Your prompt
    example_prompt=topic_prompt,
    
    # Customizations that will be added to the top and bottom of your prompt
    prefix="""You will be given a question and you have to understand to which topic it belongs. The possible topics are {topics} and you will provide as output the topic the question belongs too.
If you don't know which topics it belongs you should output 'OutOfScope'. You will be given examples to better understand how to find the right topic.""",
    suffix="Input: {question}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["question","topics"],
)


template_inh = """ You are playing the role of an expert legal assistant in the Italian Laws regulating Inheritance.
You will be given the most relevant documents related to a question and you have to examine these documents content and try to summarize an answer based on those.
Do not try to force an answer , just say the question is too complicated for you to give an answer.

You will be provided with previous messages of the conversation : 
{chat_history}

You will be also provided with some context to base your answer. This is the part you should focus on when drafting your question.
{context}

Question: {question}

Answer: """

prompt_inh = PromptTemplate(template=template_inh, input_variables=["question","context","chat_history"])


template_div = """ You are playing the role of an expert legal assistant in the Italian Laws regulating Division of Assests after Divorce.
You will be given the most relevant documents related to a question and you have to examine these documents content and try to summarize an answer based on those.
Do not try to force an answer , just say the question is too complicated for you to give an answer.

You will be provided with previous messages of the conversation : 
{chat_history}

You will be also provided with some context to base your answer. This is the part you should focus on when drafting your question.
{context}

Question: {question}

Answer: """

prompt_div = PromptTemplate(template=template_div, input_variables=["question","context","chat_history"])

memory_div = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True,input_key='question',output_key='answer',k=2)
memory_inh = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True,input_key='question',output_key='answer',k=2)

topic_chain  = LLMChain(llm=llm,prompt=topic_prompt)
divorce_question_chain = LLMChain(llm = llm , prompt=prompt_div)
inheritance_question_chain = LLMChain(llm = llm , prompt=prompt_inh)

from langchain.chains import ConversationalRetrievalChain

inheritance = ConversationalRetrievalChain.from_llm(
    llm = llm,
    chain_type='stuff',
    memory = memory_inh,
    retriever=inh_retriever,
    condense_question_llm=llm,
    return_source_documents=True,
    verbose=True)

divorce = ConversationalRetrievalChain.from_llm(
    llm = llm,
    chain_type='stuff',
    memory = memory_div,
    retriever=div_retriever,
    condense_question_llm=llm,
    return_source_documents=True,
    verbose = True)

out_examples = [
  {
    "question": "How is the weather today?",
    "answer": 
"""
I am a virtual assistant specialized in the Italian Civil Code regulating Inheritance or Division of Assests after Divorce , 
ask a question in the context of these two topics. If your question was related to such topics , please try to rephrase it.
"""
  },
  {
    "question": "Can you suggest the most beatiful cities to visit in Italy?",
    "answer": 
"""
I am a virtual assistant specialized in the Italian Civil Code regulating Inheritance or Division of Assests after Divorce , 
ask a question in the context of these two topics. If your question was related to such topics , please try to rephrase it.
"""
  },
  {
    "question": "What is the currency of Japan?",
    "answer":
"""
I am a virtual assistant specialized in the Italian Civil Code regulating Inheritance or Division of Assests after Divorce , 
ask a question in the context of these two topics. If your question was related to such topics , please try to rephrase it.
"""
  }
]

out_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Example Input: {question}\nExample Output: {answer}",
)

example_selector_out = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    out_examples, 
    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embedder, 
    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma, 
    
    # This is the number of examples to produce.
    k=3
)

out_few_prompt = FewShotPromptTemplate(
    example_selector=example_selector_out,
    example_prompt=out_prompt, 
    suffix="Question: {question}", 
    input_variables=["question"]
)

outofscope = LLMChain(llm=llm,prompt=out_few_prompt)

def run_chain(input,topics):
   topic = topic_chain({"question":input,"topics":topics})
   relevant = ""
   if "divorce" in topic["text"]:
        response =  divorce({"question":input})
        relevant = extract_source_documents(response)
   elif "inheritance" in topic["text"]:
        response = inheritance({"question":input})
        relevant = extract_source_documents(response)
   else:
        response = outofscope({"question":input})
        return response["text"]
   
   suffix = """
   Read the following laws to have a better understanding:
   """
   for rel in relevant:
       suffix = suffix + " ," + rel 
   return response["answer"] + suffix 


def get_topic(input,topics):
    return topic_chain({"question":input,"topics":topics})

def extract_source_documents(response : dict):
    source_documents = [elem.page_content for elem in response["source_documents"]]
    relevant_documents =[" ".join(elem.split()[:3]) for elem in source_documents]
    finals = list(dict.fromkeys(relevant_documents))
    print(finals)
    return finals

def extract_text(response):
    return response["text"]










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

llm = Cohere(temperature=0.7,cohere_api_key=cohere_api_key)


example_prompt = PromptTemplate(
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
    k=8
)

topic_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    
    # Your prompt
    example_prompt=example_prompt,
    
    # Customizations that will be added to the top and bottom of your prompt
    prefix="""You will be given a question and you have to understand to which topic it belongs. The possible topics are {topics} and you will provide as output the topic the question belongs too.
If you don't know which topics it belongs you should output 'OutOfScope'. You will be given examples to better understand how to find the right topic.""",
    suffix="Input: {question}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["question","topics"],
)

memory_div = ConversationBufferWindowMemory(k=3,memory_key='chat_history', return_messages=True,input_key='question')
memory_inh = ConversationBufferWindowMemory(k=3,memory_key="chat_history", return_messages=True,input_key="question")

template_inh = """ As an expert legal assistant in the Italian Laws regulating Inheritance, your goal is to provide accurate
and meaningful answers to the user questions related to this topic. You will be given context to base your answer.
Provide an answer so that a non legal expert can understand. 
Do not come up with an answer , just say you don't know if that's the case

Here you find messages of the conversation:
History : {chat_history}

Here is the context , be sure to base your answer mainly on this.
Context : {context}

Question: {question}

Answer: """

prompt_inh = PromptTemplate(template=template_inh, input_variables=["question","context","chat_history"])

template_div = """ As an expert legal assistant in the Italian Laws regulating Division of Assets after Divorce, your goal is to provide accurate
and meaningful answers to the user questions related to this topic. You will be given context to base your answer.
Provide an answer so that a non legal expert can understand. 
Do not come up with an answer , just say you don't know if that's the case

Here you find messages of the conversation:
History : {chat_history}

Here is the context , be sure to base your answer mainly on this.
Context : {context}

Question: {question}

Answer: """

prompt_div = PromptTemplate(template=template_div, input_variables=["question","context","chat_history"])


topic_chain  = LLMChain(llm=llm,prompt=topic_prompt,verbose=True)

inh_kwargs = {"prompt": prompt_inh,"memory":memory_inh}
div_kwargs = {"prompt": prompt_div,"memory":memory_div}

inheritance = RetrievalQA.from_chain_type(llm = llm,chain_type="stuff",retriever=inh_retriever,chain_type_kwargs=inh_kwargs,return_source_documents=True)

divorce = RetrievalQA.from_chain_type(llm = llm ,chain_type="stuff",retriever=div_retriever,chain_type_kwargs=div_kwargs,return_source_documents=True)

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

outofscope = LLMChain(llm=llm,prompt=out_few_prompt,verbose=True)

def run_chain(input,topics):
   topic = topic_chain({"question":input,"topics":topics})
   relevant = ""
   if "divorce" in topic["text"]:
        response =  divorce({"query":input})
        relevant = extract_source_documents(response)
   elif "inheritance" in topic["text"]:
        response = inheritance({"query":input})
        relevant = extract_source_documents(response)
   else:
        response = outofscope({"question":input})
        return response["text"]
   
   suffix = "Read the following laws to have a better understanding:"
   for rel in relevant:
       suffix = suffix + " ," + rel 
   return response["result"] + suffix


def get_topic(input,topics):
    return topic_chain({"question":input,"topics":topics})

def extract_source_documents(response : dict):
    source_documents = [elem.page_content for elem in response["source_documents"]]
    relevant_documents =[" ".join(elem.split()[:3]) for elem in source_documents]
    relevant_documents = list(dict.fromkeys(relevant_documents))
    return relevant_documents

def extract_text(response):
    return response["text"]










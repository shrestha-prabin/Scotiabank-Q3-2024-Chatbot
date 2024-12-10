import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")


LLAMA_MODEL_PATH = "app/models/llama-2-7b-chat.ggmlv3.q4_0.bin"

prompt_template = """
Use the following pieces of information to answer the user's question about Scotiabank's Q3 financial report. 
If the report does not contain the requested information, just say that you don't know and do not attempt to infer or fabricate an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Download embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initializing the Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_KEY)

index_name = "scotia"

# Loading the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model=LLAMA_MODEL_PATH,
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8},
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)


def llama_response(query: str):
    result = qa.invoke({"query": query})
    return result


llama_response("test")

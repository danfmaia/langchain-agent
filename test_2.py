from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = WebBaseLoader("https://docs.smith.langchain.com")


docs = loader.load()

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])


llm = ChatOpenAI()

output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# response = chain.invoke({"input": "how can langsmith help with testing?"})
# print(response)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# response = retrieval_chain.invoke(
#     {"input": "how can langsmith help with testing?"})
# print(response["answer"])

chat_history = [HumanMessage(
    content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response["answer"])

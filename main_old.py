from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS  # pylint: disable=E0611

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from util import output, output_uc


class AgentFunctions:

    def __init__(self):
        self.llm = ChatOpenAI()
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.output_parser = StrOutputParser()
        self.search = TavilySearchResults()

        self.documents = None
        self.vector_store = None

    def load_documents(self, docs_url):
        loader = WebBaseLoader(docs_url)
        unsplit_docs = loader.load()
        self.documents = self.text_splitter.split_documents(unsplit_docs)

    def init_vector_store(self):
        if self.documents is not None:
            self.vector_store = FAISS.from_documents(
                self.documents, self.embeddings)
        else:
            raise ValueError(
                "Documents are not loaded, cannot initialize vector store.")

    # use cases

    def simple_chain_uc(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a world-class technical documentation writer."),
            ("user", "{input}")
        ])

        chain = prompt | self.llm | self.output_parser

        response = chain.invoke(
            {"input": "how can langsmith help with testing?"})
        output(response)

    def retrieval_chain_uc(self):
        prompt = self.create_contextual_prompt()

        document_chain = create_stuff_documents_chain(self.llm, prompt)

        retriever = self.vector_store.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke(
            {"input": "how can langsmith help with testing?"})
        output(response["answer"])

    def conversation_chain_uc(self, chat_history):
        history_prompt = self.create_history_prompt()

        retriever_chain = create_history_aware_retriever(
            self.llm, self.vector_store.as_retriever(), history_prompt)

        conversation_prompt = self.create_conversation_prompt()

        document_chain = create_stuff_documents_chain(
            self.llm, conversation_prompt)

        retrieval_chain = create_retrieval_chain(
            retriever_chain, document_chain)

        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        output(response["answer"])

    def get_retriever_tool(self):
        retriever = self.vector_store.as_retriever()
        return create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )

    def new_uc(self):
        retriever_tool = self.get_retriever_tool()

        tools = [retriever_tool, self.search]

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/openai-functions-agent")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # agent_executor.invoke(
        #     {"input": "what is the weather in SF?"})

        chat_history = [HumanMessage(
            content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        agent_executor.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })

    # sample prompts

    def create_contextual_prompt(self):
        return ChatPromptTemplate.from_template(
            "Answer the following question based only on the provided context:\n"
            "<context>{context}</context>\n"
            "Question: {input}"
        )

    def create_history_prompt(self):
        return ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

    def create_conversation_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

    def simulate_chat_history(self):
        return [
            HumanMessage(
                content="Can LangSmith help test my LLM applications?"),
            AIMessage(content="Yes!")
        ]


# usage

functions = AgentFunctions()
functions.load_documents("https://docs.smith.langchain.com")
functions.init_vector_store()

# output_uc("Running simple chain UC...")
# functions.simple_chain_uc()

# output_uc("Running retrieval chain UC...")
# functions.retrieval_chain_uc()

# output_uc("Running conversation chain UC...")
# _chat_history = functions.simulate_chat_history()
# functions.conversation_chain_uc(_chat_history)

output_uc("Running new UC...")
functions.new_uc()

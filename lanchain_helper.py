import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# this file handles user inquiry -> answering questions and saving chat history

load_dotenv()
CHROMA_PATH = "./chroma"

# The vector DB only stores vectors, not the embedding model.
# The user's question must be embedded with the SAME model used in create_db.py.
embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={"allow_download": True},
)

# load the existing vector DB (this does not create or fill anything)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# retriever: finds the 3 chunks closest to the question
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM that rewrites questions and writes answers
# add a fallback if the primary_llm isn't available
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"],
    max_retries=6,
)

fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # check this name in Google's model list
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"],
)

llm = primary_llm.with_fallbacks([fallback_llm])


def format_docs(docs):
    """Join the page_content of all retrieved chunks into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


# REFORMULATE THE QUESTION
def contextualize_question():
    """Return a retriever that turns follow-up questions into standalone ones."""

    question_reformulation_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    # "system"            -> instruction for the AI model
    # MessagesPlaceholder -> filled with the messages under the key 'chat_history'
    # "human"             -> filled with the key 'input'
    template = ChatPromptTemplate.from_messages(
        [
            ("system", question_reformulation_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # if there is no history: send the raw input straight to the retriever
    # otherwise: let the LLM rewrite the question, then retrieve with the result
    return RunnableBranch(
        (lambda x: not x.get("chat_history"), (lambda x: x["input"]) | retriever),
        template | llm | StrOutputParser() | retriever,
    )


# ANSWER THE QUESTION
def answer_question():
    """Return a chain: question -> retrieve chunks -> generate answer."""

    answer_question_prompt = """Use the following pieces of retrieved context to answer the question. \
    Use three to seven sentences maximum and keep the answer concise, while still giving depth.
    If the answer is not inside the context, just say you don't know. Don't make up answers.
    Always answer in the language of the question that has been asked.

    Context: {context}"""

    template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
        ]
    )

    # join the chunks into one string, fill the prompt, call the LLM, return plain text
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | template
        | llm
        | StrOutputParser()
    )

    # step 1: retrieve chunks and store them under "context" (list of Documents)
    # step 2: generate the answer and store it under "answer"
    return RunnablePassthrough.assign(context=contextualize_question()).assign(
        answer=answer_chain
    )


# build the chain once, not on every call
rag_chain = answer_question()


# State of the conversation
class State(TypedDict):
    """
    input        : the latest user question
    chat_history : all past messages; add_messages APPENDS instead of overwriting
    context      : the chunks retrieved for the latest question
    answer       : the generated answer
    """

    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: list[Document]
    answer: str


# call the RAG chain and update the state
def call_model(state: State):
    response = rag_chain.invoke(
        {
            "input": state["input"],
            "chat_history": state.get("chat_history", []),
        }
    )

    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


# set up the workflow: START -> model -> END
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# MemorySaver keeps the history per thread_id (in RAM only)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def execute_user_query(query_text, thread_id="test-chat"):
    """
    Run the QA workflow for one question.

    query_text : the user's question
    thread_id  : identifies the conversation; same id = shared history
    returns    : the generated answer as a string
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"input": query_text}, config=config)
    return result["answer"]
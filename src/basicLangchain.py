from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Basic LLM Call ---
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)

# --- 2. Prompt Templates ---
# Instead of raw strings, LangChain uses structured templates
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains {topic} concepts clearly and concisely."),
    ("human", "{question}")
])

# --- 3. Chains (LCEL - LangChain Expression Language) ---
# The | operator pipes components together like Unix pipes
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "topic": "machine learning",
    "question": "What is gradient descent in one paragraph?"
})
print("=== Basic Chain ===")
print(response)

# --- 4. Memory / Conversation History ---
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise AI tutor."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

store = {}  # session_id -> history

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chat_prompt | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "demo-session"}}

print("\n=== Conversation with Memory ===")
r1 = conversation.invoke({"input": "My name is Alex."}, config=config)
print(f"Bot: {r1}")

r2 = conversation.invoke({"input": "What's my name?"}, config=config)
print(f"Bot: {r2}")  # It remembers!

# --- 5. Structured Output ---
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    summary: str = Field(description="One sentence summary")

parser = JsonOutputParser(pydantic_object=MovieReview)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. Respond in JSON format.\n{format_instructions}"),
    ("human", "Review the movie: {movie}")
])

review_chain = review_prompt | llm | parser

print("\n=== Structured Output ===")
review = review_chain.invoke({
    "movie": "Inception",
    "format_instructions": parser.get_format_instructions()
})
print(review)  # Returns a clean Python dict
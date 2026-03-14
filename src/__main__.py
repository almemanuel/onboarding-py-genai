from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os


load_dotenv()
history_store = {}

def get_session(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um desenvolvedor senior, líder técnico do seu time. Retorne uma crítica técnica e curta de no máximo 2 linhas."),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

reviewer_chain = reviewer_prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    reviewer_chain,
    get_session,
    input_messages_key="input",
    history_messages_key="history"
)

# invoke is stateless, so we can call it multiple times with different inputs and the context will not be shared between calls
# res = (reviewer_chain).invoke({
#     "language": "Python",
#     "code": """
#         def hello():
#             print('Hello, World!')
#     """
# })

# print(res.content)

# print("---")

# primeira chamada
res = chain_with_memory.invoke(
    {
        "input": "print('hello world')"
    },
    config={"configurable": {"session_id": "emanuel"}}
)
print(res.content)

print("---")
print("HISTÓRICO:", history_store["emanuel"].messages)
print("---")
# segunda chamada
res = chain_with_memory.invoke(
    {
        "input": "o que você achou do código anterior?"
    },
    config={"configurable": {"session_id": "emanuel"}}
)
print(res.content)

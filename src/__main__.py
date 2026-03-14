from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_agent
from langchain_core.tools import tool
import os


load_dotenv()
history_store = {}

@tool
def get_repo_info(repo_name: str) -> str: # parameter type is important for the model to understand what kind of input it should provide when calling the tool
    """Retorna informações de um repositório.""" # model read the docstring to understand what the tool does
    # mock por enquanto
    return f"Repositório {repo_name}: 42 stars, linguagem Python, última atualização hoje."

def get_session(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

agent = create_agent(
    model=llm,
    tools=[get_repo_info],
    system_prompt="Você é um assistente técnico."
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
# res = chain_with_memory.invoke(
#     {
#         "input": "print('hello world')"
#     },
#     config={"configurable": {"session_id": "emanuel"}}
# )
# print(res.content)

# print("---")
# print("HISTÓRICO:", history_store["emanuel"].messages)
# print("---")
# # segunda chamada
# res = chain_with_memory.invoke(
#     {
#         "input": "o que você achou do código anterior?"
#     },
#     config={"configurable": {"session_id": "emanuel"}}
# )
# print(res.content)

res = agent.invoke({"messages": [{"role": "user", "content": "me fala sobre o repositório langchain"}]})
print(res["messages"][-1].content)

# intern raw response with tool calls and all
for msg in res["messages"]:
    # print(msg)
    print(type(msg).__name__, ":", msg.content) # more legible output
    print("---")
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# estado do grafo - o dicionário que trafega entre os nós
# operator.add garante que as mensagens sejam acumuladas, não substituídas
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return 'end'


@tool
def get_repo_info(repo_name: str) -> str:
    """Retorna informações de um repositório."""
    # mock
    return f"Repositório {repo_name}: 42 stars, linguagem Python, última atualização hoje."


tools = [get_repo_info]

# vincula as tools ao modelo - assim ele sabe quais tem disponíveis e pode decidir chama-las
llm_with_tools = llm.bind_tools(tools)

# nó que executa a tool chamada pelo modelo, se houver
tool_node = ToolNode(tools)

# atualiza o nó call_model pra usar o modelo com tools
def call_model(state: AgentState) -> AgentState:
    res = llm_with_tools.invoke(state["messages"])
    return {"messages": [res]}


# instancia o grafo com o schema do estado
graph_builder = StateGraph(AgentState)

# adiciona um nó ao grafo
graph_builder.add_node("call_model", call_model)

graph_builder.add_node('call_tool', tool_node)

# defini as edges - START entra em call_model, call_model vai pro END ou outro ponto dependendo do edge condicional
graph_builder.add_edge(START, "call_model")

# edge condicional - se o modelo chamar uma tool, vai pra call_tool, senão termina a execução
graph_builder.add_conditional_edges(
    "call_model",
    should_use_tool,
    {
        "call_tool": 'call_tool',
        "end": END
    }
)

graph_builder.add_edge("call_tool", "call_model")

# compila o grafo - a partir daqui está pronto para executar
graph = graph_builder.compile()

if __name__ == "__main__":
    res = graph.invoke({"messages": [HumanMessage(content="Me fale sobre o repositório langgraph")]})
    
    for msg in res['messages']:
        print(type(msg).__name__, ":", msg.content)
        print("---")

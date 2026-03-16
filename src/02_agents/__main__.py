from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# tool — função Python comum decorada com @tool
# o modelo lê o docstring pra decidir quando chamar essa função
# o tipo dos parâmetros informa ao modelo o que ele precisa passar
@tool
def get_repo_info(repo_name: str) -> str:
    """Retorna informações de um repositório."""
    # mock — em produção isso seria uma chamada real à API do GitHub, por exemplo
    return f"Repositório {repo_name}: 42 stars, linguagem Python, última atualização hoje."

# cria o agente com o modelo, as tools disponíveis e a persona
# internamente o agente roda um loop:
# 1. modelo recebe a mensagem
# 2. decide se precisa chamar alguma tool
# 3. chama a tool e observa o resultado
# 4. repete até ter informação suficiente pra responder
agent = create_agent(
    model=llm,
    tools=[get_repo_info],
    system_prompt="Você é um assistente técnico."
)

if __name__ == "__main__":
    res = agent.invoke({"messages": [{"role": "user", "content": "me fala sobre o repositório langchain"}]})

    # res["messages"] contém todos os passos do raciocínio:
    # AIMessage vazio — modelo decidiu chamar uma tool
    # ToolMessage — retorno da tool
    # AIMessage com conteúdo — resposta final
    for msg in res["messages"]:
        print(type(msg).__name__, ":", msg.content)
        print("---")

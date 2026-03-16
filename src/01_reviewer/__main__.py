from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import os

load_dotenv()

# instância do modelo — é aqui que definimos qual LLM vamos usar
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# template da conversa — separa a persona (system, fixo) do input dinâmico (user)
# {history} é o placeholder onde o histórico de mensagens anteriores será injetado
# {input} é o que o usuário envia a cada chamada
reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um desenvolvedor senior, líder técnico do seu time. Retorne uma crítica técnica e curta de no máximo 2 linhas."),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

# chain — encadeia o prompt e o modelo com o operador pipe
# fluxo: formata o prompt → manda pro modelo
reviewer_chain = reviewer_prompt | llm

# dicionário que armazena o histórico de cada sessão em memória
history_store = {}

# função que retorna o histórico de uma sessão pelo id
# se a sessão não existir, cria uma nova
def get_session(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# envolve a chain com gerenciamento de histórico
# input_messages_key — campo que será salvo no histórico como mensagem do usuário
# history_messages_key — campo no template onde o histórico será injetado
chain_with_memory = RunnableWithMessageHistory(
    reviewer_chain,
    get_session,
    input_messages_key="input",
    history_messages_key="history"
)

if __name__ == "__main__":
    # primeira chamada — envia o código pra ser revisado
    # session_id identifica a conversa, igual ao id de conversa num chat
    res = chain_with_memory.invoke(
        {"input": "revise esse código:\ndef hello():\n    print('Hello, World!')"},
        config={"configurable": {"session_id": "emanuel"}}
    )
    print(res.content)

    # segunda chamada — o agente lembra do código da chamada anterior
    res = chain_with_memory.invoke(
        {"input": "o que você achou do código anterior?"},
        config={"configurable": {"session_id": "emanuel"}}
    )
    print(res.content)

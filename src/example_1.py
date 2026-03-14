from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sua-chave-aqui",
    base_url="https://api.deepseek.com"
)

resposta = llm.invoke("O que é um agente de IA?")
print(resposta.content)
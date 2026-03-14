from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

attendant_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um atendente simpático."),
    ("user", "{input}")
])

analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um analista técnico e direto."),
    ("user", "{input}")
])

attendant_chain = attendant_prompt | llm
analyst_chain = analyst_prompt | llm

res = attendant_chain.invoke({"input": "O que é um agente de IA?"})
print(res.content)

res = attendant_chain.invoke({"input": "O que é um agente de IA?"})
print(res.content)

res = analyst_chain.invoke({"input": "O que é um agente de IA?"})
print(res.content)
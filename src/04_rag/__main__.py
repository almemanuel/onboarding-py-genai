import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY')
)

# mock - em produção, seriam PDFs, páginas web, etc
documents = [
    "LangGraph é um framework para construir agentes com fluxo controlado usando grafos.",
    "ChromaDB é um banco de dados vetorial open source que roda localmente.",
    "RAG combina busca em documentos com geração de texto para respostas mais precisas.",
    "Embeddings transformam texto em vetores numéricos que representam significado semântico.",
    "LangGraph é um framework Python para construir agentes de IA com fluxo controlado usando grafos de estado."
]

# divide os documentos em chunks (pedaços menores)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents(documents)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')

#gera embeddings e armazena no ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding_model)

# retriever - busca os chunks mais relevantes para a pergunta do usuário
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda baseado apenas no contexto fornecido.:\n\n{context}"),
    ('user', "{input}")
])

if __name__ == "__main__":
    query = 'O que é LangGraph?'

    relevant_chunks = retriever.invoke(query)

    # debug - ver o que foi recuperado
    print('CHUNKS RECUPERADOS:')
    for chunk in relevant_chunks:
        print(chunk.page_content)
        print('---')

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    print('CONTEXTO ENVIADO:')
    print(context)
    print('---')

    try:
        res = (prompt | llm).invoke({
            "input": query,
            "context": context
        })
        print(res.content)
    except Exception as e:
        print(f"ERRO: {e}")
    print("FIM")
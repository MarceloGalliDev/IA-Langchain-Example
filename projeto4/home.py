# pylint: disable=all
# flake8: noqa

import os
import time
import faiss
import torch
import tempfile
import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import GCSDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

# config streamlit
st.set_page_config(page_title="Inova IA 📚", page_icon="📚")
st.header("Converse com os documentos 📚", divider='orange')

model_class = "gemini" # @param ["hf_hub", "openai", "ollama"]

model_options = {
    "vertex-google": [
        "gemini-1.5-flash-001",
        # Adicione outros modelos disponíveis para Google VertexAI
    ],
    "gemini-google": [
        "gemini-1.5-pro",
        # Adicione outros modelos disponíveis para Google VertexAI
    ],
    "hf_hub": [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "gpt2",
        "bert-base-uncased",
        # Adicione outros modelos disponíveis para Hugging Face
    ],
    "openai": [
        "gpt-3.5-turbo",
        "gpt-4",
        # Adicione outros modelos disponíveis para OpenAI
    ],
    "ollama": [
        "phi3",
        # Adicione outros modelos disponíveis para Ollama
    ],
}

# Seleção do provedor do modelo
model_class = st.sidebar.selectbox(
    "Selecione o provedor do modelo:",
    options=list(model_options.keys())
)

# Exibe apenas os modelos disponíveis para o provedor selecionado
selected_model = st.sidebar.selectbox(
    "Selecione o modelo:",
    options=model_options[model_class]
)

def model_hf_hub(model=selected_model, temperature=0.9):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 1024,
        }
    )
    return llm

def model_openai(model=selected_model, temperature=0.9):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=1024,
    )
    return llm

def model_ollama(model=selected_model, temperature=0.9):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

def model_vertex(model=selected_model, temperature=0.9):
    llm = ChatVertexAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        max_retries=6,
        timeout=None,
        stop=None,
    )
    return llm

def model_google(model=selected_model, temperature=0.9):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        max_retries=6,
        stop=None,
        timeout=None
    )
    return llm

# Função para configurar o LLM com base na seleção do usuário
def get_llm(model_class, model_name):
    if model_class == "hf_hub":
        return model_hf_hub(model=model_name)
    elif model_class == "openai":
        return model_openai(model=model_name)
    elif model_class == "ollama":
        return model_ollama(model=model_name)
    elif model_class == "vertex-google":
        return model_vertex(model=model_name)
    elif model_class == "gemini-google":
        return model_google(model=model_name)

# Inicialize o modelo com base nas seleções
llm = get_llm(model_class, selected_model)

def config_retriever(uploads):
    docs = []
    temp_dir = tempfile.mkdtemp()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir, file.name)
        # acessando a pasta temp
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        # carregando o arquivo pdf
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # divisao em pedaços os textos/split
    text_splitter = RecursiveCharacterTextSplitter(
        # vamos criar 1000 documentos
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # embeddings, igualmente ao openia
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # vetores de armazenamento
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # configuração do retriever, influencia nas analises e desempenho
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    return retriever


def config_retriever_google():
    docs = []
    temp_dir = tempfile.mkdtemp()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir, file.name)
        # acessando a pasta temp
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        # carregando o arquivo pdf
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # divisao em pedaços os textos/split
    text_splitter = RecursiveCharacterTextSplitter(
        # vamos criar 1000 documentos
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # embeddings, igualmente ao openia
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # vetores de armazenamento
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # configuração do retriever, influencia nas analises e desempenho
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    return retriever


def config_rag_chain(model_class, retriever):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    elif model_class == "gemini":
        llm = model_vertex()

    # definição de prompt
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # prompt de contextualização
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question question which can be understood without the chat history, Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    # adicionando o token de inicio
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt)
        ]
    )
    
    # chain contextualização
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_q_prompt
    )

    qa_prompt_template = """Você é um atendente escrivão especialista.
    Você sempre fala fatos nas fontes fornecidas e nunca inventa fatos novos.
    Você so responde sobre perguntas pertinentes a cartórios de todos os tipos.
    Você sempre retorna uma mensagem concisa.
    Você consulta dados de documentos indicados, podendo ser arquivos pdf, txt ou consultas ao big query.
    Utilize o contexto para responder as perguntas do usuário, caso não tenha contexto, análise a parte da documentação indica para iniciar o contexto.
    Caso não consiga responder ou gerar um resposta, informe que não conseguiu.
    Agora, olhe para esses dados e responda às perguntas. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# painel lateral
uploads = st.sidebar.file_uploader(
    label="Enviar arquivos",
    type=["pdf"],
    accept_multiple_files=True
)
if not uploads:
    st.info("Por favor, envie algum arquivo para continuar")
    st.stop()

# variáveis de sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou seu assistente virtual! Como posso ajuda-lo?")
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# conversação
start = time.time()
user_query = st.chat_input("Digite sua mensagem...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    # atualizamos os documentos
    with st.chat_message("AI"):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)

        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke(
            {
                "input": user_query,
                "chat_history": st.session_state.chat_history
            }
        )

        resp = result['answer']
        st.write(resp)

        # mostrar fonte
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            ref = f":link: Fonte {idx}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)
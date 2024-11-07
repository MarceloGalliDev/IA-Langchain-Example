# pylint: disable=all
# flake8: noqa

import os
import tempfile
import streamlit as st
from time import sleep
from pathlib import Path
from fake_useragent import UserAgent
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
    LLMChain
)

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {
    'Groq': {
        'modelos': [
            'llama-3.1-70b-versatile',
            'gemma2-9b-it',
            'mixtral-8x7b-32768'
        ],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': [
            'gpt-4o-mini',
            'gpt-4o',
        ],
        'chat': ChatOpenAI
    }
}

RETRIEVAL_SEARCH_TYPE = "mmr"
RETRIEVAL_ARGS = {"k": 5, "fetch_k": 20}
PROMPT = """
Você é um atendente escrivão especialista.
Você sempre se apega aos fatos nas fontes fornecidas
e nunca inventa fatos novos.
Você so responde sobre perguntas pertinentes a cartórios de todos os tipos.
Você sempre retorna uma mensagem bem formatada.
Você consulta dados de documentos indicados, pois estamos em ambiente seguro
e com pessoas autenticadas.
Utilize o contexto para responder as perguntas do usuário.
Agora, olhe para esses dados e responda às perguntas.

Contexto:
{context}

Conversa atual:
{chat_history}
Human: {question}
Assistant: """

MEMORIA = ConversationBufferMemory()


def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento


def carrega_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao carregar o site {i+1}')
            sleep(3)
    if documento == '':
        st.error('Não foi possível carregar o site')
        st.stop()
    return documento


def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def input_personalizado_tipo_arquivo():
    tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
    if tipo_arquivo == 'Site':
        arquivo = st.text_input('Digite a url do site')
    if tipo_arquivo == 'Pdf':
        arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
    if tipo_arquivo == 'Csv':
        arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
    if tipo_arquivo == 'Txt':
        arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
    return arquivo


def input_personalizado_modelo_ia():
    provedor = st.selectbox('Selecione o provedor do modelo', CONFIG_MODELOS.keys())
    modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
    return provedor, modelo

def importacao_arquivos():
    temp_dir = st.session_state['temp_dir']
    documentos = []
    for arquivo in Path(temp_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(arquivo))
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos

def split_de_documentos(documentos):
    recursive_spliter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""]
    )

    documentos = recursive_spliter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata['source'].split('/')[1]
        doc.metadata['doc_id'] = i

    return documentos

def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store

def cria_chain_conversa(provedor, modelo):
    documentos = importacao_arquivos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)


    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )

    retriever = vector_store.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs=RETRIEVAL_ARGS
    )

    chat_class = CONFIG_MODELOS[provedor]['chat']
    chat = chat_class(model_name=modelo)

    prompt_template = ChatPromptTemplate.from_messages([
        ('system', get_config('prompt')),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    question_generator_chain = LLMChain(llm=chat, prompt=prompt_template)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt_template},
        question_generator=question_generator_chain
    )

    st.session_state['chain'] = chat_chain

def get_config(config_name):
    if config_name.lower() in st.session_state:
        return st.session_state[config_name.lower()]
    elif config_name.lower() == 'prompt':
        return PROMPT
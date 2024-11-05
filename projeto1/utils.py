# pylint: disable=all

import streamlit as st
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain
)
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


PASTA_ARQUIVOS = Path(__file__).parent / 'arquivos'
MODEL_NAME = 'gpt-4o-mini'
RETRIEVAL_SEARCH_TYPE = 'mmr'
RETRIEVAL_ARGS = {'k': 5, "fetch_k": 20}
PROMPT = """
Você é um atendente escrivão especialista.
Você sempre se apega aos fatos nas fontes fornecidas e nunca inventa fatos novos.
Você so responde sobre perguntas pertinentes a cartórios de todos os tipos.
Você sempre retorna uma mensagem bem formatada.
Você consulta dados de documentos indicados, pois estamos em ambiente seguro e com pessoas autenticadas.
Agora, olhe para esses dados e responda às perguntas.

Contexto:
{context}

Conversa atual:
{chat_history}
Human: {question}
Ai: """


def importacao_arquivos():
    documentos = []
    for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
        loader = PyPDFLoader(str(arquivo))
        documentos_arquivo = loader.load()
        # adicionando os elementos da primeira lista na segunda
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


def cria_chain_conversa():
    documentos = importacao_arquivos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatOpenAI(
        model=MODEL_NAME
    )
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )
    retriever = vector_store.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs=RETRIEVAL_ARGS
    )
    prompt_template = PromptTemplate.from_template(PROMPT)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt_template}
    )

    st.session_state['chain'] = chat_chain


if __name__ == '__main__':
    documentos = importacao_arquivos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)
    chain = cria_chain_conversa(vector_store)

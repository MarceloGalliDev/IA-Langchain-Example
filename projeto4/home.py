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
from langchain_core.output_parsers import StrOutputParser
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

# Configuração do Streamlit
st.set_page_config(page_title="Inova IA 📚", page_icon="📚")
st.header("Converse com os documentos 📚", divider='orange')

model_class = "openai"  # @param ["hf_hub", "openai", "ollama"]

model_options = {
    "openai": [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        # Adicione outros modelos disponíveis para OpenAI
    ],
    "gemini": [
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
    ],
    "hf_hub": [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "gpt2",
        "bert-base-uncased",
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

def model_hf_hub(model=selected_model, temperature=0.5):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 1024,
        }
    )
    return llm

def model_openai(model=selected_model, temperature=0.5):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=1024,
    )
    return llm

def model_ollama(model=selected_model, temperature=0.5):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

def model_vertex(model=selected_model, temperature=0.5):
    llm = ChatVertexAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        max_retries=6,
        timeout=None,
        stop=None,
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
    elif model_class == "gemini":
        return model_vertex(model=model_name)

# Inicialize o modelo com base nas seleções
llm = get_llm(model_class, selected_model)

def config_retriever(uploads):
    if not uploads:
        return None

    docs = []
    temp_dir = tempfile.mkdtemp()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir, file.name)
        # Acessando a pasta temp
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        # Carregando o arquivo PDF
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Divisão em pedaços os textos/split
    text_splitter = RecursiveCharacterTextSplitter(
        # Vamos criar 1000 documentos
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    if not splits:
        st.warning("Nenhum texto encontrado nos documentos carregados.")
        return None

    # Embeddings, igualmente ao OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Vetores de armazenamento
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Configuração do retriever, influencia nas análises e desempenho
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

    # Definição de prompt
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # Prompt de contextualização
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    # Adicionando o token de início
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt)
        ]
    )
    
    # Chain de contextualização
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_q_prompt
    )

    qa_prompt_template = """Você é um atendente escrivão especialista.
    Você sempre fala fatos nas fontes fornecidas e nunca inventa fatos novos.
    Você só responde sobre perguntas pertinentes a cartórios de todos os tipos e sobre casos que envolvem direito, como leis e afins.
    Você sempre retorna uma mensagem concisa.
    Você pode consultar dados de documentos indicados, podendo ser arquivos pdf, txt ou consultas ao BigQuery.
    Utilize o contexto para responder as perguntas do usuário, caso não tenha contexto, analise a parte da documentação indicada para iniciar o contexto.
    Caso não consiga responder ou gerar uma resposta, informe que não conseguiu.
    Você sempre responde no idioma português do Brasil.
    Agora, olhe para esses dados e responda às perguntas. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# Função para configurar o chat sem RAG
def config_simple_chat(model_class):
    def model_chat(user_query, chat_history, model_class):
        # Inicialize o modelo apropriado
        if model_class == "hf_hub":
            llm = model_hf_hub()
        elif model_class == "openai":
            llm = model_openai()
        elif model_class == "ollama":
            llm = model_ollama()
        elif model_class == "gemini":
            llm = model_vertex()

        system_prompt = """
            Você é um atendente escrivão especialista.
            Você sempre fala fatos nas fontes fornecidas e nunca inventa fatos novos.
            Você só responde sobre perguntas pertinentes a cartórios de todos os tipos e sobre casos que envolvem direito, como leis e afins.
            Você sempre responde no idioma português do Brasil.
        """
        
        if model_class.startswith("hf"):
            user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            user_prompt = "{input}"

        # Cria o template de prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  # Usado para armazenar o histórico das conversas
            ("user", user_prompt)
        ])
        
        chain = prompt_template | llm | StrOutputParser()
        
        return chain.stream({
            "chat_history": chat_history,
            "input": user_query
        })

    return model_chat

# Painel lateral - botão upload
uploads = st.sidebar.file_uploader(
    label="Enviar arquivos",
    type=["pdf"],
    accept_multiple_files=True
)

# Variáveis de sessão
# Conferindo se existe sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou seu assistente virtual! Como posso ajuda-lo?")
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "last_uploads" not in st.session_state:
    st.session_state.last_uploads = []

# Histórico de mensagens
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Conversação
start = time.time()
user_query = st.chat_input("Digite sua mensagem...")

# Função para formatar o tempo
def format_elapsed_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Atualizamos os documentos se houver novos uploads
    with st.chat_message("AI"):
        # Detecta se houve novos uploads
        if uploads and st.session_state.last_uploads != uploads:
            st.session_state.last_uploads = uploads
            st.session_state.retriever = config_retriever(uploads)
            st.session_state.docs_list = uploads

            if st.session_state.retriever:
                rag_chain = config_rag_chain(model_class, st.session_state.retriever)
                result = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = result['answer']
                resp = st.write(answer)
                
                sources = result['context']
                for idx, doc in enumerate(sources):
                    source = doc.metadata['source']
                    file = os.path.basename(source)
                    page = doc.metadata.get('page', 'Página não especificada')

                    ref = f":link: Fonte {idx}: *{file} - p. {page}*"
                    with st.popover(ref):
                        st.caption(doc.page_content)

                # Adiciona a resposta ao histórico
                st.session_state.chat_history.append(AIMessage(content=answer))
            else:
                # Fallback caso não haja retriever
                resp = "Documentos carregados, mas houve um problema ao processá-los."
        else:
            # Sem novos uploads, verifica se já existe um retriever configurado
            if st.session_state.retriever:
                rag_chain = config_rag_chain(model_class, st.session_state.retriever)
                result = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = result['answer']
                resp = st.write(answer)

                sources = result['context']
                for idx, doc in enumerate(sources):
                    source = doc.metadata['source']
                    file = os.path.basename(source)
                    page = doc.metadata.get('page', 'Página não especificada')

                    ref = f":link: Fonte {idx}: *{file} - p. {page}*"
                    with st.popover(ref):
                        st.caption(doc.page_content)

                # Adiciona a resposta ao histórico
                st.session_state.chat_history.append(AIMessage(content=answer))
            else:
                # Sem retriever, realiza chat normal
                chat_function = config_simple_chat(model_class)
                resp = chat_function(user_query, st.session_state.chat_history, model_class)
                # Exibe a resposta
                full_join = st.write_stream(resp)
                full_response = "".join(full_join)
                st.session_state.chat_history.append(AIMessage(content=full_response))

    # Medir e exibir o tempo de resposta
    end = time.time()
    elapsed_time = end - start
    formatted_time = format_elapsed_time(elapsed_time)
    st.write(f"Tempo: {formatted_time}")

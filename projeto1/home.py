# pylint: disable=all

import requests
import base64
import streamlit as st
import tempfile
from pathlib import Path
from utils import cria_chain_conversa
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def load_files_to_session(uploaded_files, temp_dir):
    """Loads uploaded files into session state and saves them to a temporary directory."""
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = {}

    for pdf in uploaded_files:
        file_content = pdf.read()
        # Store file content in session state for persistence across reruns
        st.session_state['uploaded_files'][pdf.name] = file_content
        # Also write to temp_dir for any additional processing that requires a file path
        file_path = Path(temp_dir) / pdf.name
        with open(file_path, 'wb') as f:
            f.write(file_content)


def sidebar():
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    temp_dir = st.session_state['temp_dir']

    upload_pdfs = st.file_uploader(
        'Adicione aqui seus arquivos!',
        type=['.pdf'],
        accept_multiple_files=True
    )
    if upload_pdfs is not None:
        load_files_to_session(upload_pdfs, temp_dir)

    solicitacoes = ['Selecione uma solicitação', 'solicitacao001', 'solicitacao002', 'solicitacao003']

    if 'codigo_solicitacao' not in st.session_state:
        st.session_state['codigo_solicitacao'] = 'Selecione uma solicitação'

    # Obter o índice da seleção atual
    if st.session_state['codigo_solicitacao'] in solicitacoes:
        current_index = solicitacoes.index(st.session_state['codigo_solicitacao'])
    else:
        current_index = 0  # Posição de 'Selecione uma solicitação'

    # Selecionar a solicitação usando a mesma chave
    codigo_solicitacao = st.selectbox(
        'Selecione a Solicitação',
        solicitacoes,
        index=current_index,
        key='codigo_solicitacao'
    )

    # Check for changes in the selected request
    if codigo_solicitacao != 'Selecione uma solicitação':
        if 'last_codigo_solicitacao' not in st.session_state or \
            st.session_state['last_codigo_solicitacao'] != codigo_solicitacao:
            st.session_state['last_codigo_solicitacao'] = codigo_solicitacao

            # Clear any files currently in session_state['uploaded_files'] and temp_dir
            st.session_state['uploaded_files'].clear()
            for arquivo in Path(temp_dir).glob('*'):
                arquivo.unlink()

            # Fetch PDFs from the API
            data = {
                'bucket_name': 'langchain-inova',
                'codigo_cartorio': 'cartorio001',
                'codigo_solicitacao': codigo_solicitacao,
            }
            response = requests.post(
                'https://apiia.inovacartorios.com.br/api/v1/gcp/get_pdfs/',
                json=data
            )
            if response.status_code == 200:
                pdf_files = response.json()

                for pdf in pdf_files:
                    pdf_content = base64.b64decode(pdf['content'])
                    st.session_state['uploaded_files'][pdf['filename']] = pdf_content
                    file_path = Path(temp_dir) / pdf['filename']
                    with open(file_path, 'wb') as f:
                        f.write(pdf_content)

                st.success('Documentos carregados com sucesso!')
                st.write('Documentos disponíveis:')
                for pdf in pdf_files:
                    st.write(f"- {pdf['filename']}")

                cria_chain_conversa()
            else:
                st.error('Não foi possível obter os documentos.')
    else:
        st.write('Por favor, selecione uma solicitação.')

    label_botao = 'Inicializar IA' if 'chain' not in st.session_state else 'Atualizar IA'
    if st.button(label_botao, use_container_width=True):
        if len(st.session_state['uploaded_files']) == 0:
            st.error('Adicione um arquivo .pdf para inicializar o chat!')
        else:
            st.success('Inicializando o chat...')
            cria_chain_conversa()


def chat_window():
    st.header('Bem-Vindo à IA do Inova Cartórios', divider='orange')

    if 'chain' not in st.session_state:
        st.error('Faça upload de PDF para inicializar!')
        st.stop()

    chain = st.session_state['chain']
    memory = chain.memory

    mensagens = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input('Digite aqui o Prompt...')
    if nova_mensagem:
        chat = container.chat_message('human')
        chat.markdown(nova_mensagem)
        chat = container.chat_message('ai')
        chat.markdown('Gerando resposta...')

        resposta = chain({'question': nova_mensagem})
        st.session_state['ultima_resposta'] = resposta['answer']


def main():
    with st.sidebar:
        sidebar()
    chat_window()


if __name__ == '__main__':
    main()

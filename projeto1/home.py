# pylint: disable=all

import streamlit as st
import time
from utils import cria_chain_conversa, PASTA_ARQUIVOS

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def sidebar():
    upload_pdfs = st.file_uploader(
        'Adicione aqui seus arquivos!',
        type=['.pdf'],
        accept_multiple_files=True
    )

    # listando todos pdf dentro dessa pasta
    if upload_pdfs is not None:
        for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
            # deletando arquivos
            arquivo.unlink()
        for pdf in upload_pdfs:
            with open(PASTA_ARQUIVOS / pdf.name, 'wb') as f:
                f.write(pdf.read())

    # alterando o text do button
    label_botao = 'Inicializar IA'
    if 'chain' in st.session_state:
        label_botao = 'Atualizar IA'
    if st.button(label_botao, use_container_width=True):
        if len(list(PASTA_ARQUIVOS.glob('*.pdf'))) == 0:
            st.error('Adicione o arquivo .pdf para inicializar o chat!')
        else:
            st.success('Inicializando o chat...')
            cria_chain_conversa()
            st.rerun()


def chat_window():
    st.header('Bem-Vindo a IA do Inova Cartórios', divider='orange')

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

        chain.invoke({'question': nova_mensagem})
        time.sleep(2)
        st.rerun()


def main():
    with st.sidebar:
        sidebar()

    chat_window()


if __name__ == '__main__':
    main()

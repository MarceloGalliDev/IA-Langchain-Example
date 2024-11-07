# pylint: disable=all
# flake8: noqa


import streamlit as st

def pagina_chat():
    st.header('🤖Bem-vindo ao IAinova', divider='orange')


def sidebar():
    st.button('Sessão', use_container_width=True)


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
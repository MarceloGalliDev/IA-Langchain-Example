# pylint: disable=all
# flake8: noqa

import streamlit as st

MODEL_NAME = "gpt-4o-mini"
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


def get_config(config_name):
    if config_name.lower() in st.session_state:
        return st.session_state[config_name.lower()]
    elif config_name.lower() == 'model_name':
        return MODEL_NAME
    elif config_name.lower() == 'retrieval_search_type':
        return RETRIEVAL_SEARCH_TYPE
    elif config_name.lower() == 'retrieval_kwargs':
        return RETRIEVAL_ARGS
    elif config_name.lower() == 'prompt':
        return PROMPT

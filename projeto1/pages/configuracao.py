# pylint: disable=all
# flake8: noqa

import sys
import json
from pathlib import Path
import streamlit as st
sys.path.append(str(Path(__file__).resolve().parent.parent))
from projeto1.configs import (
    get_config
)
from projeto1.utils import PASTA_ARQUIVOS, cria_chain_conversa

def config_page():
    st.header('Página de configuração', divider='orange')

    model_name = st.text_input(
        'Tipo de Modelo',
        value=get_config('model_name')
    )
    retrieval_search_type = st.text_input(
        'Tipo de Retriever',
        value=get_config('retrieval_search_type')
    )
    retrieval_kwargs = st.text_input(
        'Tipo de Paramêtros Retriever',
        value=json.dumps(get_config('retrieval_kwargs'))
    )
    prompt = st.text_area(
        'Contexto da IA',
        height=500,
        value=get_config('prompt')
    )

    if st.button('Salvar Parâmetros', use_container_width=True):
        retrieval_kwargs = json.loads(retrieval_kwargs.replace("'", '"'))
        st.session_state['model_name'] = model_name
        st.session_state['retrieval_search_type'] = retrieval_search_type
        st.session_state['retrieval_kwargs'] = retrieval_kwargs
        st.session_state['prompt'] = prompt
        st.rerun()

    if st.button('Atualizar IA', use_container_width=True):
        if len(list(PASTA_ARQUIVOS.glob('*.pdf'))) == 0:
            st.error('Adicione o arquivo .pdf para inicializar o chat!')
        else:
            st.success('Inicializando o chat...')
            cria_chain_conversa()
            st.rerun()


config_page()

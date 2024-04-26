# RAG_Streamlit_llama2

## create new env 
python -m venv env_name

## activate the env
env_name/bin/activate **(Mac/Linux)**

env_name\Scripts\activate **(Windows)**

## install the requirements
pip install -r requirements.txt

## download the model from hugging-face: llama-2-7b-chat.ggmlv3.q4_0.bin
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

## run the app
streamlit run streamlit_app.py

import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os

@st.cache_resource
def carrega_modelo():
    url = 'https://drive.google.com/uc?id=1w5doSoVqMcmMIdGHgrVlGeoDvmHxZtYL'
    output = 'modelo_mosquito_final.tflite'

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    if not os.path.exists(output):
        raise FileNotFoundError(f"Arquivo {output} não encontrado após download.")

    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagens():
    uploaded_files = st.file_uploader(
        'Arraste e solte **uma ou mais imagens** aqui ou clique para selecionar',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    imagens = []
    nomes = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = image.resize((224, 224))
            st.image(image, caption=uploaded_file.name, width=150)

            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)

            imagens.append(image_array)
            nomes.append(uploaded_file.name)

        st.success(f"{len(imagens)} imagem(ns) carregada(s) com sucesso")
        return imagens, nomes

    return None, None

def previsao(interpreter, imagem):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], imagem)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = [
        'Culex_landing',
        'Culex_smashed',
        'aegypti_landing',
        'aegypti_smashed',
        'albopictus_landing',
        'albopictus_smashed'
    ]

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]

    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade por classe'
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))

    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica mosquito na pele humana",
        page_icon="🦟",
    )

    st.title("Classifica mosquito na pele humana! 🦟")

    # Carrega modelo
    interpreter = carrega_modelo()

    # Carrega imagens
    imagens, nomes = carrega_imagens()

    # Classifica
    if imagens:
        for imagem, nome in zip(imagens, nomes):
            st.subheader(f"🖼️ Resultado para: {nome}")
            previsao(interpreter, imagem)

if __name__ == "__main__":
    main()

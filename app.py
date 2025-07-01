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
    url = 'https://drive.google.com/uc?id=13q14tlNUio1yoiMvXHQMBs9x9HrnfmJY'
    output = 'modelo_mosquitos.tflite'

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    if not os.path.exists(output):
        raise FileNotFoundError(f"Arquivo {output} não encontrado após download.")

    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image = image.resize((224, 224)) 
        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

        return image

def previsao(interpreter, imagem):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], imagem)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # DEBUG: veja se o modelo está retornando alguma coisa
    st.write("Saída do modelo:", output_data)

    classes = ['Culex_landing',
    'Culex_smashed',
    'aegypti_landing',
    'aegypti_smashed',
    'albopictus_landing',
    'albopictus_smashed']
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]

    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade de Classes'
    )

    st.plotly_chart(fig)  

def main():
    st.set_page_config(
        page_title="Classifica mosquito na pele humana",
        page_icon="🦟",

    )

    st.write("# Classifica mosquito na pele humana! 🦟")


    #Carrega modelo
    interpreter = carrega_modelo()

    #Carrega imagem
    image= carrega_imagem()

    #Classifica
    if image is not None:
        
        previsao(interpreter,image)

if __name__ == "__main__":
    main()
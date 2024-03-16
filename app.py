from helper_functions import *

llm = OpenAI(temperature=0)

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

import os
from flask import Flask, request, redirect, jsonify, url_for, session, abort
from flask_cors import CORS

# ########## INIT APP ##########

# --- API Flask app ---
app = Flask(__name__)
app.secret_key = "super secret key"

CORS(app)

# --- BACKEND API ---
@app.route("/")
# @app.doc(hide=True)
def index():
    """Define the content of the main fontend page of the API server"""

    return f"""
    <h1>The 'Omdena's IREX API' server is up.</h1>
    """


@app.route("/inference", methods=["POST", "GET"])
def inference():
    """Infer using the LLM model and MDFEND"""
    
    news = """
    General Cienfuegos furioso :" Si queremos un cambio debemos revelarnos contra EPN"
    En un vídeo difundido en redes sociales por la cadena de noticias "Cierre de Edición" se confirma la participación de miembros del Ejercito Mexicanos, quienes planean rebelarse contra el presidente de México Enrique Peña Nieto.
    El Ejercito Nacional encabezado por el General Salvador Cienfuegos Zepeda, titular de la Secretaría de la Defensa Nacional (Sedena), reconoció que la entrada de los militares a la lucha contra el narcotráfico fue un error. El Ejército, dijo, debió resolver un problema "que no nos tocaba", debido a que las corporaciones estaban corrompidas y esta acción solo les permitió ampliar la ventaja para seguir con sus actos ilícitos.
    Sumando la condición en la que se encuentra actualmente México con el aumento de la gasolina el General Cienfuegos afirmo que el ejército está a la entera y completa disposición del pueblo mexicano, "Creo que no me toca decirlo a mí, pero las encuestas lo dicen, este gobierno está perdido, solo falta la decisión del pueblo y el ejercito los respaldara y protegerá para hacer cumplir sus derechos constitucionales." asimismo añadió "Si queremos un cambio vamos a rebelarnos en contra de Peña Nieto
    """

    if request.method == "POST":
        query = request.form.get("query")
        path = "sample.json"
        if True:
            dummy = {}
            dummy["text"] = "dummy text"
            dummy["domain"] = 8
            dummy["label"] = 0

            user_data = {}
            user_data["text"] = query
            user_data["domain"] = 8
            user_data["label"] = 0

            user_data = pd.DataFrame([dummy, user_data])
            user_data.to_json(path)
            # path = "sample.json"
            tokenizer = TokenizerFromPreTrained(max_len, bert)
            verification_process = ProcessPipeline(llm, news , path , tokenizer)
            result = verification_process.process_news(use_filter=False)
            return result
    else:
        return f"""
    <h1>Inference server</h1>
    """


# ########## START FLASK SERVER ##########

if __name__ == "__main__":

    current_port = int(os.environ.get("PORT") or 5000)
    app.debug = True
    app.run(debug=False, host="0.0.0.0", port=current_port, threaded=True)

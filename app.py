from flask import Flask, render_template,request, jsonify
import pandas as pd
from pypdf import PdfReader
from qdrant_client import models, QdrantClient
import os
from dotenv import load_dotenv, dotenv_values
from chat import vector_db, chat_bot


load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

quad = vector_db()
# quad.create_collection()
chatbot= chat_bot()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get("message")
    response = chatbot.get_chat(user_input)
    print(response)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

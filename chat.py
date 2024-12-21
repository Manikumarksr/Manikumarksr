import pandas as pd
from pypdf import PdfReader
from qdrant_client import models, QdrantClient
import aisuite as ai
import os
from dotenv import load_dotenv, dotenv_values
from sentence_transformers import SentenceTransformer
import re
import markdown

load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')


class vector_db(QdrantClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qdrant = QdrantClient(":memory:") 
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def process_pdf(self, pdf:str):
        reader = PdfReader(pdf) 
        page = reader.pages[0] 
        text=""
        for i in range(len(reader.pages)): 
            page = reader.pages[i]
            text+=page.extract_text()
        sentences = text.split(".") # Split text into sentences
        return sentences

    def create_collection(self, pdf='static/Manikumar_CV.pdf', collection_name="mk_cv"):
        sentences = self.process_pdf(pdf)
        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        self.qdrant.upload_points(
            collection_name=collection_name,
            points=[models.PointStruct(
                    id=idx,
                    vector=self.encoder.encode(sentence),  # Encode sentence to vector
                    payload={"content":sentence},) for idx, sentence in enumerate(sentences) ])

    def search(self,query:str,collection_name="mk_cv"):
        hits = self.qdrant.search(
        collection_name=collection_name,
        query_vector=self.encoder.encode(query).tolist(),
        limit=20)
        return [hit.payload for hit in hits]

# quad = vector_db()
# quad.create_collection()
# res= quad.search("AI/ML Research associate role")
# print(res)
# create the vector database client

class chat_bot():
    def __init__(self):
        self.vector_db = vector_db()
        self.vector_db.create_collection()
        self.history = [
            {"role": "system", "content":f"You are a professional assistant representing ManiKumar R, a highly skilled Master's student at the Indian Institute of Science with expertise in Machine Learning (ML), Deep Learning (DL), and their applications in biological research. Manikumar also has significant experience in quantitative research, financial markets, algorithmic trading, IoT, large language models (LLMs), and web development. Your primary goal is to answer any queries about Manikumar's professional background, skills, projects, and expertise based on their CV."},
           
            ]

    def search_in_pdf(self, query:str):
        hits= self.vector_db.search(query)
        content= [hit['content'] for hit in hits]
        content= ". ".join(content)
        return content
    
    def format_response(self, response: str) -> str:
        # Convert Markdown to HTML
        # formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)  # Bold
        # formatted_response = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_response)  # Italic
        # formatted_response = formatted_response.replace("\n", "<br>")  # Line breaks
        formatted_response = markdown.markdown(response)
        return formatted_response

    def get_chat(self, user_prompt:str):
        self.history.append({"role": "user", "content": user_prompt})
        context = self.search_in_pdf(user_prompt)
        self.history.append({"role": "system", "content": f"Additional context form CV: {context}"})
        client = ai.Client()
        response = client.chat.completions.create(model="groq:llama-3.1-8b-instant", messages=self.history)
        response = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": response})
        return self.format_response(response)




# chatbot= chat_bot()
# # chatbot = ai.Client()
# print("done")
# print(chatbot.get_chat("What are your skills"))

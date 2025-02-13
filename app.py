from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)  # Initialize Flask app

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medibot"

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding = embeddings
    )
retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GEMINI_API_KEY
)

# Create the chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({
        "input":msg, 
        "system_prompt": system_prompt})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)
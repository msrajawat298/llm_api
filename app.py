from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import logging

# Set the logging level to ignore warnings from the sentence_transformers module
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially google Palm api key)

app = Flask(__name__)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="deepset/roberta-large-squad2")
vectordb_file_path = os.environ["VECTOR_DB_FILE_PATH"]

def create_vector_db():
    # Specify the encoding as 'latin-1'
    file_encoding = 'latin-1'
    loader = CSVLoader(file_path=os.environ["TRAIN_DATA_FILE_PATH"], source_column="prompt", encoding=file_encoding)
    # Load data from FAQ sheet
    data = loader.load()
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know. you can email your query to my team on email msrajawat298@gmail.com " Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

@app.route('/answer', methods=['POST'])
def answer_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Question field is required"}), 400
    chain = get_qa_chain()
    answer = chain(question)
    return jsonify({"answer": answer})

@app.route('/', methods=['GET'])
def dummy_get():
    return jsonify({"message": "This is a dummy GET request!"})

if __name__ == "__main__":
    create_vector_db()
    app.run(debug=True)

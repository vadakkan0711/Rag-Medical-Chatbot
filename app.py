from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import re
from dotenv import load_dotenv
import os
import pickle



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"



# Load Existing index 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)



contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def clean_response(response):
    """
    Cleans the response by removing leading newlines, 'AI:' prefixes, and extra spaces.
    """
    # Remove "AI:" prefix with any leading spaces or newlines
    return re.sub(r"^\s*AI:\s*", "", response).strip()




conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



# Load the diabetes model
diabetes_model_path = os.path.join('models', 'diabetes_model.sav')
with open(diabetes_model_path, 'rb') as file:
    diabetes_model = pickle.load(file)

# Load the heart disease model
heart_model_path = os.path.join('models', 'heart_disease_model.sav')
with open(heart_model_path, 'rb') as file:
    heart_disease_model = pickle.load(file)

# Load the Parkinson's disease model
parkinsons_model_path = os.path.join('models', 'parkinsons_model.sav')
with open(parkinsons_model_path, 'rb') as file:
    parkinsons_model = pickle.load(file)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat")
def chatroute():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = conversational_rag_chain.invoke(
    {"input": msg},
    config={"configurable": {"session_id": "abc123"}},
    )["answer"]
    cleaned_response = clean_response(response)
    print("Response : ",cleaned_response)
    return str(cleaned_response)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None  # Initialize result to None for GET requests

    if request.method == 'POST':
        # Collect form data from the request
        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        SkinThickness = request.form.get('SkinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')

        try:
            # Convert inputs to float and create the feature array
            user_input = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]

            # Predict using the loaded model
            prediction = diabetes_model.predict([user_input])

            # Map the prediction result to a human-readable message
            if prediction[0] == 1:
                result = ("Our analysis suggests a higher likelihood of diabetes."
                "Please consult a healthcare professional for a detailed diagnosis and further guidance.")
            else:
                result = ("Our analysis suggests no immediate signs of diabetes. "
              "However, maintaining a healthy lifestyle is always beneficial for long-term well-being.")

        except ValueError:
            # Handle errors if the input cannot be converted to float
            result = "Invalid input! Please enter numeric values for all fields."

    return render_template('diabetes.html', result=result)



@app.route('/cardiac', methods=['GET', 'POST'])
def heart():
    result = None  # Initialize result for GET requests

    if request.method == 'POST':
        # Collect form data
        age = request.form.get('age')
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')

        try:
            # Create input array
            user_input = [
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope), float(ca), float(thal)
            ]

            # Predict using the heart disease model
            prediction = heart_disease_model.predict([user_input])

            # User-friendly response
            if prediction[0] == 1:
                result = ("Our analysis suggests a higher likelihood of heart disease. "
                          "Please consult a cardiologist for a detailed evaluation.")
            else:
                result = ("Our analysis suggests no immediate signs of heart disease. "
                          "However, regular check-ups and a heart-healthy lifestyle are always beneficial.")

        except ValueError:
            result = "Invalid input! Please enter numeric values for all fields."

    return render_template('cardiac.html', result=result)


@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = None  # Initialize result for GET requests

    if request.method == 'POST':
        # Collect form data
        inputs = [
            request.form.get('fo'), request.form.get('fhi'), request.form.get('flo'),
            request.form.get('Jitter_percent'), request.form.get('Jitter_Abs'),
            request.form.get('RAP'), request.form.get('PPQ'), request.form.get('DDP'),
            request.form.get('Shimmer'), request.form.get('Shimmer_dB'), request.form.get('APQ3'),
            request.form.get('APQ5'), request.form.get('APQ'), request.form.get('DDA'),
            request.form.get('NHR'), request.form.get('HNR'), request.form.get('RPDE'),
            request.form.get('DFA'), request.form.get('spread1'), request.form.get('spread2'),
            request.form.get('D2'), request.form.get('PPE')
        ]

        try:
            # Convert inputs to float
            user_input = [float(x) for x in inputs]

            # Predict using the Parkinson's disease model
            prediction = parkinsons_model.predict([user_input])

            # User-friendly response
            if prediction[0] == 1:
                result = ("Our analysis suggests a higher likelihood of Parkinson's disease. "
                          "Please consult a neurologist for further evaluation.")
            else:
                result = ("Our analysis suggests no immediate signs of Parkinson's disease. "
                          "However, regular monitoring and a healthy lifestyle are recommended.")

        except ValueError:
            result = "Invalid input! Please enter numeric values for all fields."

    return render_template('parkinsons.html', result=result)





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
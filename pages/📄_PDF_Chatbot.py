import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
# import image_ocr
import easyocr
import numpy as np
import pdf2image
from io import BytesIO
import glob
import time

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

image_link=[]

try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}

#EXTRACT TEXT
def get_file_text_from_local():
    text = ""
    reader = easyocr.Reader(['en'])
    pdf_files = list(Path('G:\My Drive\pdf-sets').rglob('*.pdf'))  # Recursively find all PDF files

    for file_path in pdf_files:
        file = Path(file_path)
        st.write(f"Processing file: {file.name}")  # Debug output to show which file is being processed 
        text += f"\n--- The text below is from {file.name} ---\n"
        check = False
        try:
            # Reading the PDF file
            with file.open('rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
                        check = True
        except Exception as e:
            st.write(f"Error reading text from {file.name}: {e}")  # Debug output for error handling

        # If text extraction from PDF fails, try OCR
        if not check:
            try:
                file_bytes = file.read_bytes()
                images = pdf2image.convert_from_bytes(file_bytes)
                for page in images:
                    results = reader.readtext(np.array(page))
                    for i in results:
                        text += i[1] + " "
            except Exception as e:
                st.write(f"Error processing OCR for {file.name}: {e}")  # Debug output for error handling
    
    return text


# VECTORISASI
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(r"G:\My Drive\faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question in a detailed and expressive manner, providing as much relevant information from the provided context as possible. Make sure the response is storytelling and engaging. 

    If the context is not provided or the context is empty:
    Answer like a normal AI chatbot with a storytelling effect.
    
    If the answer is not in the provided context:
    Answer like a normal chatbot if there is no context provided and specify that the answer is not in the provided context.  
    
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    # "The question you asked is not available in the context, did you mean ... ?" or you could
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local(r"G:\My Drive\faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])  # Equal column width for compact fit
    
    # Add custom CSS for button styling
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if button_cols[0].button("Yes"):
        clear_chat_history()
        st.rerun()
    elif button_cols[1].button("No"):
        st.rerun()
        
def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('gemini_history', None)
    
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Sidebar Menu')
    
    if st.session_state.get('chat_id') is None:
        # st.write(past_chats.keys())
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
        # st.write(st.session_state)
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    if st.button("Clear Chat History", key="clear_chat_button"):
        # st.write(st.session_state)
        modal()
    
    
    st.session_state.chat_title = f'PDF-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.write('# Chat with Gemini')


# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

# Check if 'messages' is not in st.session_state and initialize with a default message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "✨",  # or any valid emoji
        "content": "Hey there, I'm your Text Extraction chatbot. Please upload the necessary files in the sidebar to add more context to this conversation."
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.markdown(message['content'])


if prompt := st.chat_input('Your message here...'):
    # Display user message in chat message container
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )


    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt, api_key)
        # output = ""
        # for i, element in enumerate(check_docs):
        #     output += f"Element {i}:\n{element}\n\n"
       
        # pdf_files = list(Path('G:\My Drive\pdf-sets').rglob('*.pdf'))
        # for file in pdf_files:
        #     if file.name not in output:
        #         start_time = time.time()
        #         raw_text = get_file_text_from_local()
        #         st.write("Extracted text:")
        #         st.write(raw_text)
        #         #end timer
        #         text_chunks = get_text_chunks(raw_text)
        #         get_vector_store(text_chunks, api_key)
        #         elapsed_time = time.time() - start_time
        #         minutes, seconds = divmod(elapsed_time, 60)
        #         st.write(f"Processing took {int(minutes)} minute(s) and {seconds:.2f} seconds.")

            
        
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=response,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

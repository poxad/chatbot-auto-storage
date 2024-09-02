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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
import easyocr
import numpy as np
import pdf2image
from io import BytesIO
import streamlit as st
import pyodbc
import pandas as pd
from datetime import date
import re
import matplotlib.pyplot as plt
import ast
# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )

conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        column_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return column_names, rows

# Load environment variables from a .env file
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)


try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}



# -------------------FUNCTIONS DEF---------------------
def code_generator(user_question,companies,chat_history,column_names):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    prompt = f"""
    column names: {column_names}
    column Company: {companies}

    The user will provide a question or command, and your job is to convert it into a SQL query. 
    1. The SQL query should only be a SELECT statement from the 'dbo.data' table.
    2. Use `TOP` for limiting results when asking for the first rows.
    3. For the last rows, use `ORDER BY <YourColumn> DESC` and `TOP`.
    4. For specific row ranges, use `OFFSET` and `FETCH NEXT` with `ORDER BY <YourColumn>`.
       - Ensure <YourColumn> is a valid column in the 'dbo.data' table.
    5. The user is allowed to request operations like SELECT, DISPLAY, SUM, AVG, and similar read-only operations.
    6. Ensure that the query cannot modify data (e.g., no INSERT, UPDATE, DELETE).
    7. Only query using column name from the table.
    8. Do not use the SUBSTR function

    After generating the query, use pandas' read_sql function to run the query, which will return a dataframe. Here is an example on how to use it:
    - pd.read_sql(the SQL query generated, conn)
    
    - Wrap the dataframe in st.write(),e.g. st.write(pd.read_sql(the SQL query generated, conn)).
    - Else, if user dont ask to display it, then don't.

    Example: 
    user_question: display top 10 rows
    output should be : st.write(pd.read_sql("SELECT TOP 10 * FROM dbo.data", conn))
    
    user_question: create bar chart to compare order_total for each company
    output should be : chart_data = pd.read_sql("SELECT Company, SUM(Order_Total) AS Total_Order FROM dbo.data GROUP BY Company", conn); st.bar_chart(chart_data, x="Company", y="Total_Order")

    user_question: create bar chart to compare sum of order_total and received_total for each company
    output should be : chart_data = pd.read_sql("SELECT Company, SUM(Order_Total) AS Total_Order, SUM(Received_Total) AS Total_Received FROM dbo.data GROUP BY Company", conn); st.bar_chart(chart_data, x="Company", y="Total_Order"); st.bar_chart(chart_data, x="Company", y="Total_Received")

    Else If the user asks to plot a graph, such as a bar chart, line chart, or any other type of graph:
    - Generate a Python code that uses the DataFrame generated before for plotting.
    - Use `.groupby('Company')` if the data requires aggregation by a specific column, such as summing `Order_Total` and `Received_Total` for each company.
    - After aggregating, use Streamlit's chart elements to plot out the data.
    - Some Streamlit chart elements you can use are:
    - `st.bar_chart()`: Displays a bar chart. Use it for visualizing data in bar format. Parameters include `data`, `x`, `y`, and `color`.
    - `st.line_chart()`: Displays a line chart. Ideal for showing trends over time. Parameters include `data`, `x`, `y`, and `color`.
    - `st.area_chart()`: Displays an area chart. Useful for showing cumulative totals or trends with shaded areas. Parameters include `data`, `x`, `y`, and `color`.
    - `st.scatter_chart()`: Displays a scatter plot. Best for visualizing the relationship between two continuous variables. Parameters include `data`, `x`, `y`, and `color`.
    - `st.map()`: Displays a geographical map. Ideal for visualizing location data. Parameters include `data` with latitude and longitude columns.
    - When using streamlit's chart elements, make sure to include the x axis and y axis inside the function. For e.g. st.bar_chart(dataframe_name,x="x_axis_column",y="y_axis_column")
    - The result of the code should only be 1 line. Combine multiple lines of code into 1 line using ";". For example: chart_data = pd.read_sql("the SQL query generated above", conn).groupby('Company').sum();st.bar_chart(chart_data)
    - Do not include comments and import package code in the final result.
    - Extract the relevant data from the context and calculate the sum directly.

    Below is an example on how you should generate the script code, in this case, user's question is : "Create a barchart that shows the sum of order_total and received_total of each company, name it Total_Order."
    - You will generate a code similar like the one below and you should use this format of code.
    - chart_data = pd.read_sql("the SQL query generated above", conn).groupby('Company')[['Order_Total', 'Received_Total']].sum();chart_data['Total_Orders'] = chart_data['Order_Total'] + chart_data['Received_Total']; st.bar_chart(chart_data[['Total_Orders']])


    User input: {user_question}
    """

    #    Please explain the output of the chart or table you generated.
    # You can add this line of code: ;st.write("your explanation of the contents inside the chart or table")
    response = llm.invoke(f"Given the following conversation history:\n{chat_history}\n {prompt}")
    return response.content

def     generate_explanation_with_ai(response_code, conn):
    sql_query_match = re.search(r'pd.read_sql\("(.*?)", conn\)', response_code)
    if sql_query_match:
        sql_query = sql_query_match.group(1)
        df = pd.read_sql(sql_query, conn)
        df_string = df.to_string(index=False)

        prompt = f"""
        I have the following data from a SQL query:

        {df_string}

        Provide a detailed explanation of this data.
        """
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        explanation_response = llm.invoke(prompt)
        return explanation_response.content.strip()

    return "Could not generate an explanation. Check the code or SQL query."


@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])  # Equal column width for compact fit
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

def clean_response(response):
    response = response.strip("```sql\n").strip("```").strip()
    response = response.replace("\n", " ")
    response = ' '.join(response.split())
    return response

def clean_response_python(response):
    if('```' in response):
        response = response.strip("```python\n").strip("```").strip()
        end_index = response.find('```')
        if end_index != -1:
            response = response[:end_index].strip()

    else:
        parts = response.split('\n\n')  # Assuming code and unwanted text are separated by new lines
        code_part = parts[0].strip()
        response=code_part
    
    return response

def df_to_string(df):
    # Convert DataFrame to a string format that can be used as context
    context = df.to_string(index=False)
    return context


# --------------------SIDEBAR------------------------------
with st.sidebar:
    st.write('# Sidebar Menu')

    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
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

    
    st.session_state.chat_title = f'Database-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

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
        "content": "Hey there, I'm your Database Chatbot. Ask me to display rows to be used as the context for this conversation. e.g. Display the top 10 rows"
    })

# new_chat_id = f'{time.time()}'

# if 'current_chat_id' not in st.session_state or st.session_state['current_chat_id'] != new_chat_id:
#     # Reset chat states
#     st.session_state['current_chat_id'] = new_chat_id
#     st.session_state['messages'] = []
#     st.session_state['gemini_history'] = []
    
#     # Start a new chat session
#     st.session_state.chat_title = f'Database-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
#     st.session_state.chat_id = new_chat_id
#     st.session_state.model = genai.GenerativeModel('gemini-pro')
#     st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.write(message['content'])
        if(message['role']== MODEL_ROLE):
            try:
                exec(message['content'])
            except Exception as e:
                st.write(e)

# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    # st.write(past_chats.keys()) #ngecheck chat history punya id, berapa chat IDs yang di stored buat history
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    if 'data_table' not in st.session_state:
        st.session_state.data_table = None
    if 'column_names' not in st.session_state:
        st.session_state.column_names = None
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])

    with st.spinner("Waiting for AI response..."):
        df=pd.read_sql("SELECT distinct Company FROM dbo.data", conn)
        companies = df_to_string(df['Company'])
        
        df2=pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'data'", conn)
        column_names = df_to_string(df2['column_name'])


        response=code_generator(prompt,companies,chat_history,column_names)
        explanation = generate_explanation_with_ai(response, conn)
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):  
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=response,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.write(response)
        try:
            exec(response)
        except Exception as e:
            st.write(e)
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=explanation,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.write(explanation)
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

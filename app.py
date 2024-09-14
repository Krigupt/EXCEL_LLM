import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import time
from sqlalchemy import create_engine, MetaData, Table, Column, String,inspect
from google.cloud.sql.connector import Connector
import pymysql
import os
from dotenv import load_dotenv
import json
# Load environment variables from .env file
load_dotenv()

# Set Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  "/Users/krishna/Desktop/LLM_EXCEL/mysql-435509-7225659bfe4d.json"
import os
import certifi




from langchain_community.llms import GooglePalm
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = certifi.where()
# Initialize Cloud SQL Connector
connector = Connector()

def getconn() -> pymysql.connections.Connection:
    conn = connector.connect(
        os.getenv("DB_INSTANCE"),
        "pymysql",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        db=os.getenv("DB_NAME")
    )
    return conn

# Create SQLAlchemy engine
engine = create_engine("mysql+pymysql://", creator=getconn)

# Define the scope for Google Sheets API
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Load the credentials for Google Sheets
creds = Credentials.from_service_account_file(
    'sheets-435420-01db2ff57fc4.json', scopes=scope
)




client = gspread.authorize(creds)        
        
# Function to create MySQL table using SQLAlchemy
def create_table(table_name, columns):
    metadata = MetaData()
    columns_def = [Column(col, String(225)) for col in columns]
    table = Table(table_name, metadata, *columns_def, extend_existing=True)
    
    with engine.connect() as conn:
        metadata.create_all(conn)
        st.write(f"Table '{table_name}' created successfully!")







# Function to insert data into MySQL using SQLAlchemy
def insert_data(table_name, columns, data):
    metadata = MetaData()
    
    table = Table(table_name, metadata, autoload_with=engine)
   
    with engine.connect() as conn:
        data_dict = [dict(zip(columns, row)) for row in data]
        conn.execute(table.insert(), data_dict)
        conn.commit()
        st.write(f"Data inserted into table '{table_name}'.")






# Retry wrapper function
def retry_request(func, *args, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func(*args)
        except (gspread.exceptions.APIError, Exception) as e:
            st.error(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Failed after {retries} retries.")


# Function to retrieve all table names from the database
def get_table_names():
    inspector = inspect(engine)
    return inspector.get_table_names()


# Streamlit App Code
st.title("Google Sheets to MySQL")

with st.sidebar:
    st.header("Add Google Sheets ID")
    
    # Form to input Google Sheets ID
    with st.form(key='google_sheets_form'):
        sheet_id = st.text_input("Google Sheets ID")
        submit_button = st.form_submit_button("Add Table from Sheet")
        
        if submit_button:
            if sheet_id:
                # Fetch Google Sheets data and insert it into MySQL
                try:
                    spreadsheet = retry_request(client.open_by_key, sheet_id)
                    
                    # Process each worksheet in the Google Sheet
                    for worksheet in spreadsheet.worksheets():
                        sheet_name = worksheet.title
                        st.write(f"Processing sheet: {sheet_name} from Google Sheet ID: {sheet_id}")
                        
                        # Get data from Google Sheets
                        data = worksheet.get_all_values()
                        
                        
                        # Convert to DataFrame (first row as columns, rest as data)
                        df = pd.DataFrame(data[1:], columns=data[0])
                        
                        
                        # Clean column names
                        df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_') for col in df.columns]
                        
                        
                        # Generate table name based on sheet name and part of Google Sheets ID
                        table_name = f"{sheet_name}".replace(' ', '_').replace('-', '_')
                        
                        # Create table in MySQL
                        create_table(table_name, df.columns)
                        
                        # Insert data into the MySQL table
                        insert_data(table_name, df.columns, df.values)
                        
                        st.success(f"Data from {sheet_name} has been added to the table '{table_name}' in the database.")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a valid Google Sheets ID.")

# Display current tables in the database
st.sidebar.subheader("Current Tables in Database")
tables = get_table_names()
if tables:
    for table in tables:
        st.sidebar.write(table)
else:
    st.sidebar.write("No tables found.")






# Initialize the LLM
llm = ChatGroq(
    groq_api_key="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",
    model_name="Llama3-8b-8192",
    streaming=True
    
)




# Create SQLDatabase object using the engine
db = SQLDatabase(engine, sample_rows_in_table_info=100)
toolkit=SQLDatabaseToolkit(db=db,llm=llm)


agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        def update_response():
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

        update_response()


from sqlalchemy import create_engine, text
import os
import click
from datetime import datetime
from dotenv import load_dotenv
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import random

# Load environment variables
load_dotenv()

# Setup database connection
DATABASE_URL = os.getenv('DATABASE_URL')  # Connection string for SQL database
engine = create_engine(DATABASE_URL)

# Define column names for each table
MEDICATION_COLUMNS = "['id', 'PatientID', 'MedicationName', 'Dosage', 'Frequency', 'StartDate', 'EndDate']"
PATIENT_COLUMNS = "['id', 'Name', 'PhoneNumber', 'Email', 'DOB']"
TODAY_TIME = datetime.now().isoformat()
MAX_FEEDBACK_ATTEMPTS = 3
POSITIVE_EXAMPLES = []
RESPONSE = ""
FEEDBACK_FILE = "feedback_history.json"


def load_feedback_history():
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_feedback_history(examples):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(examples, f, indent=2)


POSITIVE_EXAMPLES = load_feedback_history()

# Example preprocessing function to clean the user input
def preprocess_input(user_input):
    user_input = user_input.strip().capitalize()
    return user_input

# Create a prompt for the SQL query
def create_prompt(user_query):
    db_schema = f"""
    The database has two tables:
    1. patients table with columns: {PATIENT_COLUMNS}
    2. medications table with columns: {MEDICATION_COLUMNS}
    
    Important: Return the query.
    Example valid responses:
    'medications SELECT id, MedicationName, Dosage FROM medications WHERE PatientID = 1'
    'patients SELECT id, Name, PhoneNumber FROM patients WHERE Name = "John"'
    """
    
    return f"{db_schema}\n\nConvert this question into a SQL query: {user_query}"

def get_sql_query_from_llm(prompt):
    global RESPONSE
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    messages = [
        {"role": "system", "content": """You are a healthcare query generator and assistant. 

If the user asks questions unrelated to healthcare medications or patient information, respond with:
OFFTOPIC: followed by a helpful message explaining that you can only answer questions about healthcare medications and patients. Include 1-2 example questions they could ask instead.

For valid, supported questions, follow these rules:
- Always prefix queries with the table name followed by '/'
- Use standard SQL keywords: SELECT, FROM, WHERE, ORDER BY, etc.
- Always specify exact columns to select, never use SELECT *
Example valid queries:
'SELECT id, MedicationName, Dosage FROM medications WHERE id = 1'
'SELECT id, Name, PhoneNumber FROM patients WHERE Name = "John"'
"""},

        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "OFFTOPIC: I can only help with questions about healthcare medications and patient information. Try asking questions like:\n- What medications is patient John Smith taking?\n- Can you find patient Nicole Roy's contact information?"},
        {"role": "user", "content": "Show me all medications for patient 1"},
        {"role": "assistant", "content": f"SELECT id, MedicationName, Dosage, Frequency FROM medications WHERE PatientID = 1"},
        {"role": "user", "content": "Find patient Nicole Roy"},
        {"role": "assistant", "content": "SELECT id, Name, PhoneNumber FROM patients WHERE Name = 'Nicole Roy'"},
    ]
    
    # Add positive examples to messages
    for example in POSITIVE_EXAMPLES:
        messages.extend([
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example.get("sql_query", example["answer"])}
        ])
    
    # Add current query
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    response = completion.choices[0].message.content.strip()
    RESPONSE = response
    
    # Handle off-topic and unsupported responses
    if response.startswith('OFFTOPIC:'):
        print("\n" + response[9:].strip()) 
        return None
    elif response.startswith('UNSUPPORTED:'):
        print("\n" + response[12:].strip())  
        return None
    #print(f"\nGenerated SQL Query: {response}\n")
    return response


def execute_sql_query(sql_query):
    with engine.connect() as connection:
        result = connection.execute(text(sql_query))
        return [dict(row) for row in result]

def post_processing(results, clean_query):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are an expert at formatting data into human readable formats. Return the answer in a human readable format rather than a JSON"},
            {"role": "system", "content": "Depending on the query, you may not need to perform any data logic but instead just return the answer in a human readable format as the logic will have already been done in the query."},
            {"role": "system", "content": "Remove any prefix information like Here is your answer in a human readable format. Simply return the answer."},
            {"role": "assistant", "content": "For example, if given results: {'Name': 'Jack Dougherty'} {'Name': 'Kenneth Castillo'} and query 'Give me a list of patients that came in last week', I would return: 'The patients who came in for an appointment last week are: Jack Dougherty and Kenneth Castillo.'"},
            {"role": "assistant", "content": "For the same results but query 'How many patients do I have named Jack', I would return: 'You have one patient named Jack: Jack Dougherty.'"},
            {"role": "user", "content": f"Return this data in a human readable format: {results}\n\n For your context, the original query was: {clean_query} "}
    ]
    )
    
    return completion.choices[0].message.content.strip()

# Command Line Interface
@click.group()
def cli():
    """
    Healthcare Assistant - Your AI-powered healthcare management helper!
    
    This tool helps you manage healthcare data by answering questions about
    medications and patients using natural language. Simply ask questions like
    you would ask a human assistant.
    
    Commands:
        ask     Ask questions about medications and patients
        explain Learn how the assistant processes your questions
    """
    pass

def collect_feedback():
    attempts = 0
    while attempts < MAX_FEEDBACK_ATTEMPTS:
        response = input("\nDo you feel this answer is helpful? (Y/N): ").strip().upper()
        if response == 'Y':
            return True
        elif response == 'N':
            return False
        else:
            print("Please answer with the provided choices (Y/N)")
            attempts += 1
    return None

# Verbose output option
VERBOSE = False

def verbose_print(message):
    if VERBOSE:
        print(f"\n{message}")

@cli.command()
@click.argument('question')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def ask(question, verbose):
    """Ask questions about medications and patients."""
    global POSITIVE_EXAMPLES, RESPONSE, VERBOSE
    VERBOSE = verbose
    print("Getting you your answer...\n")
    
    verbose_print("Preprocessing user input...")
    clean_query = preprocess_input(question)
    
    verbose_print("Creating prompt for LLM...")
    prompt = create_prompt(clean_query)
    
    verbose_print("Generating SQL query from LLM...")
    sql_query = get_sql_query_from_llm(prompt)
    
    # Stop here if the query was off-topic (sql_query will be None)
    if sql_query is None:
        return
    
    verbose_print("Executing query against SQL database...")
    results = execute_sql_query(sql_query)
    
    verbose_print("Post-processing results...")
    answer = post_processing(results, clean_query)
    print(f"\nAnswer: {answer}")
    
    if random.random() < 0.3:
        verbose_print("Collecting feedback...")
        feedback = collect_feedback()
        if feedback:
            POSITIVE_EXAMPLES.append({
                "question": question,
                "answer": RESPONSE
            })
            save_feedback_history(POSITIVE_EXAMPLES)
            print("Thank you for your feedback! Your feedback directly helps improve the models.")

@cli.command()
def explain():
    """Learn how the healthcare assistant processes your questions."""
    explanation = """
    How the Healthcare Assistant Works
    ===================================
    
    Information Flow
    ----------------
    1. You ask a question in plain English
    2. The assistant processes your question through several steps:
       a. Preprocessing to clean and standardize the input
       b. Converting to a SQL query using AI
       c. Executing the query against the database
       d. Converting the results back to natural language
    """
    print(explanation)

if __name__ == '__main__':
    cli()






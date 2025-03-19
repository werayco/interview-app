import random
import warnings
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def load_questions():
    with open("questions.txt", "r") as questions_file:
        return [line.strip() for line in questions_file.readlines()]

interview_questions = load_questions()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
groq_api_key = os.getenv("GROQ_API_KEY")
warnings.filterwarnings("ignore")

# Initialize ChatGroq
chat_bot = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key, temperature=0.5, max_tokens=500)

st.title("AI Interview Preparation Bot")

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def ask_question():
    question = random.choice(interview_questions)
    st.session_state.current_question = question
    memory.save_context({"interview_status": "started"}, {"question": question})
    return question

def analyze_response(user_response: str):
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    last_question = st.session_state.current_question
    
    template = """
    You are an AI assistant helping with interview preparation.
    Your task is to:
    1. Analyze the user's response for clarity, relevance, and impact.
    2. Provide constructive feedback.

    Interview Question: {last_question}
    User Response: {user_response}

    Feedback:
    - Strengths of the response
    - Areas for improvement
    - Suggestions for a stronger answer
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    final_prompt = prompt.format(last_question=last_question, user_response=user_response)
    response = chat_bot.invoke(final_prompt).content
    
    memory.save_context({"question": last_question}, {"answer": user_response})
    st.session_state.chat_history.append((last_question, user_response, response))
    return response

if not st.session_state.interview_started:
    if st.button("Start Interview"):  
        st.session_state.interview_started = True
        st.session_state.current_question = ask_question()
        st.write(f"**Interview Question:** {st.session_state.current_question}")
else:
    st.write(f"**Interview Question:** {st.session_state.current_question}")
    user_answer = st.text_input("Your Answer:")
    
    if st.button("Submit Answer") and user_answer:
        feedback = analyze_response(user_answer)
        st.write("**Feedback:**")
        st.write(feedback)
        
        if st.button("Next Question"):
            st.session_state.current_question = ask_question()
            st.write(f"**Next Interview Question:** {st.session_state.current_question}")

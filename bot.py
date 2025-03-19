import streamlit as st
import random
import warnings
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

groq_api_key = st.secrets["GROQ"]
warnings.filterwarnings("ignore")

with open("questions.txt", "r") as questions:
    interview_questions = [line.strip() for line in questions.readlines()]

# Initialize memory and chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_bot = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key, temperature=0.5, max_tokens=500)

def analyze_response(user_response: str, last_question: str):
    template = """
    You are an AI assistant helping with interview preparation.
    Your task is to:
    1. Analyze the user's response for clarity, relevance, and impact.
    2. Provide constructive feedback that directly relates to the interview question.

    Interview Question: {last_question}
    User Response: {user_response}

    Feedback:
    - Strengths of the response
    - Areas for improvement
    - Suggestions for a stronger answer that is relevant to the question
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_prompt = prompt.format(last_question=last_question, user_response=user_response)
    response = chat_bot.invoke(final_prompt).content
    memory.save_context({"question": last_question}, {"answer": user_response})
    return response

st.title("🗣️ Interview Chatbot")
st.write("Welcome to your AI-powered interview preparation. Answer questions and receive feedback in real time!")

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.interview_started:
    if st.button("🎤 Start Interview"): 
        st.session_state.interview_started = True
        st.session_state.question = random.choice(interview_questions)
        memory.save_context({"interview_status": "started"}, {"question": st.session_state.question})
        st.session_state.chat_history.append(("bot", st.session_state.question))
        st.rerun()
else:
    chat_container = st.container()
    with chat_container:
        for role, text in st.session_state.chat_history:
            if role == "bot":
                st.chat_message("assistant").write(text)
            elif role == "user":
                st.chat_message("user").write(text)
            else:
                st.write(f"**Feedback:** {text}")
    
    user_answer = st.chat_input("Type your response here...")
    if user_answer:
        last_question = st.session_state.question
        st.session_state.chat_history.append(("user", user_answer))
        feedback = analyze_response(user_answer, last_question)
        st.session_state.chat_history.append(("bot", feedback))
        st.session_state.question = random.choice(interview_questions)
        memory.save_context({"question": st.session_state.question}, {"answer": user_answer})
        st.session_state.chat_history.append(("bot", st.session_state.question))
        st.rerun()
    
    if st.button("❌ End Interview"):
        st.session_state.interview_started = False
        st.session_state.chat_history = []
        st.write("Interview session ended. Good luck!")

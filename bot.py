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

# Load interview questions
with open("questions.txt", "r") as questions:
    interview_questions = [line.strip() for line in questions.readlines()]

# Initialize memory and chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_bot = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key, temperature=0.5, max_tokens=500)

def analyze_response(user_response: str):
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    last_question = "Unknown question"
    for message in reversed(chat_history):
        if "question" in str(message): 
            last_question = str(message.content)
            break

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
    return response

# Streamlit UI
st.title("Interview Preparation Assistant")
st.write("Practice answering interview questions and receive AI-generated feedback.")

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.interview_started:
    if st.button("Start Interview"): 
        st.session_state.interview_started = True
        st.session_state.question = random.choice(interview_questions)
        memory.save_context({"interview_status": "started"}, {"question": st.session_state.question})
        st.session_state.chat_history.append(("Question", st.session_state.question))
        st.rerun()
else:
    st.write(f"**Interview Question:** {st.session_state.question}")
    user_answer = st.text_area("Your Answer:")
    
    if st.button("Submit Answer"):
        feedback = analyze_response(user_answer)
        st.session_state.chat_history.append(("Answer", user_answer))
        st.session_state.chat_history.append(("Feedback", feedback))
        st.session_state.question = random.choice(interview_questions)
        memory.save_context({"question": st.session_state.question}, {"answer": user_answer})
        st.rerun()
    
    st.write("### Chat History")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")
    
    if st.button("End Interview"):
        st.session_state.interview_started = False
        st.session_state.chat_history = []
        st.write("Interview session ended. Good luck!")

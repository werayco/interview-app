import random
import warnings
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()

# Define interview questions
interview_questions = [
    "Tell me something about yourself.",
    "Can you please introduce yourself?",
    "Can you walk me through your resume and highlight your experience relevant to this position?",
    "What are your strengths?",
    "What are your weaknesses?",
    "What are your accomplishments?",
    "Have you ever made any mistakes? How did you handle them?",
    "Why did you leave your last position? Why are you leaving your current position?",
    "How would your supervisor/colleagues describe you?",
    "How would you rate your last employer?",
    "What kind of manager would you prefer to work with?",
    "Do you think you are overqualified for this position?",
    "Do you have any plans for the next five years?",
    "Where do you see yourself in the next five years?",
    "What’s your long-term career goal?",
    "How do you know about us?",
    "What do you know about us?",
    "Why are you interested in this position?",
    "Why do you want to work for us?",
    "Why should we hire you?",
    "Give an example of when you had to work with someone who was difficult to get along with. How did you handle it?",
    "Tell me about a time when you were communicating with someone, and they did not understand you. What did you do?",
    "Give me an example of a time you faced a conflict while working on a team. How did you handle it?",
    "Describe a time when you struggled to build a relationship with someone important. How did you overcome that?",
    "Tell me about one of your favorite experiences working with a team and your contribution.",
    "Describe the best partner or supervisor you’ve worked with. What part of their managing style appealed to you?",
    "Can you share an experience where a project dramatically shifted directions at the last minute? What did you do?",
    "We all make mistakes we wish we could take back. Tell me about a time you wish you’d handled a situation differently with a colleague.",
    "Tell me about a time you needed to get information from someone who wasn’t very responsive. What did you do?",
    "Tell me about a time when you failed in a team project and how you overcame it.",
    "Tell me about a time you had to be very strategic to meet all your top priorities.",
    "Tell me about a time when you had to juggle several projects at the same time. How did you organize your time? What was the result?",
    "Sometimes it’s just not possible to get everything on your to-do list done. Tell me about a time your responsibilities got overwhelming. What did you do?",
    "Tell me about a project that you planned. How did you organize and schedule the tasks?",
    "Give an example of a time when you delegated an important task successfully.",
    "How would you go about simplifying a complex issue to explain it to a client or colleague?",
    "Give me an example of a time when you were able to successfully persuade someone to see things your way at work.",
    "Tell me about a successful presentation you gave and why you think it did well.",
    "Tell me about a time when you had to rely on written communication to get your ideas across to your team.",
    "Give me an example of when you had to explain something fairly complex to a frustrated client. How did you handle it?",
    "Describe a time when you saw a problem and took the initiative to correct it rather than waiting for someone else.",
    "Tell me about a time when you worked with minimal supervision. How did you handle that?",
    "Give me an example of a time you were able to be creative with your work. What was exciting or challenging about it?",
    "Tell me about a time you were dissatisfied with your work. What could have been done to make it better?",
    "Share an example of how you were able to motivate a coworker, your peers, or your team.",
    "Tell me about a time when you were asked to do something you had never done before. How did you react? What did you learn?",
    "Tell me about the biggest change that you have had to deal with. How did you adapt?",
    "Describe a situation in which you embraced a new system, process, technology, or idea at work.",
    "Recall a time when you were assigned a task outside of your job description. How did you handle it? What was the outcome?",
    "Tell me about a time you were under a lot of pressure. How did you get through it?",
    "Describe a time when it was vital to make a good impression on a client. How did you go about it?",
    "Give me an example of a time when you did not meet a client’s expectations. What happened, and how did you attempt to rectify the situation?",
    "Tell me about a time when you made sure a customer was pleased with your service.",
    "Describe a situation where you needed to persuade someone to see things your way. What steps did you take? What were the results?",
    "Tell me about the toughest decision you had to make.",
    "Have you ever had to 'sell' an idea to your coworkers or group? How did you do it? What were the results?",
    "Tell me about a time when you handled a challenging situation.",
    "What’s your salary expectation?",
    "Do you mind working overtime/relocating?",
    "When can you start if we hire you?",
    "How do you spend your spare time?",
    "Do you have any references?",
    "Do you have any questions for us?",
    "Give a specific example of how you have demonstrated your ability to handle multiple priorities and deliver results.",
    "Tell me about a decision that you made objectively, despite having personal opinions.",
    "Tell me about a time when you have had to manage or resolve a conflict between two (or more) co-workers.",
    "Give an example which demonstrates your ability to develop successful working relationships.",
    "Give me a specific occasion when you conformed to a policy which you did not agree with.",
    "Tell me about a time when you were faced with a stressful situation and needed to use your coping skills.",
    "Tell me about a time when you were not satisfied with your own performance. What did you do about it?",
    "Describe when you have been given negative feedback, and explain how you handled this.",
    "Tell me about a time when you initiated a procedural change. How did you present these ideas to management?",
    "Describe a time when you were recognized for dealing effectively with a difficult situation.",
    "Describe a problem which you needed to resolve in your last job. What was the impact of this?"
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

groq_api_key = st.secrets["GROQ"]
chat_bot = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key, temperature=0.1, max_tokens=500)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def ask_question():
  return random.choice(interview_questions)

def analyze_response(question, user_response):
    template = """
    You are an AI assistant helping with interview preparation.
    Your task is to:
    1. Analyze the user's response for clarity, relevance, and impact.
    2. Provide constructive feedback.

    Interview Question: {question}
    User Response: {user_response}

    Feedback:
    - Strengths of the response
    - Areas for improvement
    - Suggestions for a stronger answer
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_prompt = prompt.format(question=question, user_response=user_response)
    response = chat_bot.invoke(final_prompt).content
    return response

st.title("AI-Powered Interview Coach")
st.write("Prepare for your next job interview with AI-generated feedback.")

if "question" not in st.session_state:
    st.session_state.question = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "user_response" not in st.session_state:
    st.session_state.user_response = ""

if st.button("Start Interview") or st.session_state.question:
    if st.session_state.question is None:
        st.session_state.question = ask_question()
    
    st.subheader("Interview Question")
    st.write(st.session_state.question)
    
    st.session_state.user_response = st.text_area("Your Response", st.session_state.user_response)
    
    if st.button("Submit Response"):
        if st.session_state.user_response.strip():
            st.session_state.feedback = analyze_response(st.session_state.question, st.session_state.user_response)
        else:
            st.warning("Please enter a response before submitting.")
    
    if st.session_state.feedback:
        st.subheader("AI Feedback")
        st.write(st.session_state.feedback)
    
    if st.button("Next Question"):
        st.session_state.question = ask_question()
        st.session_state.feedback = None
        st.session_state.user_response = ""

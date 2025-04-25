# Importing Necessary Dependencies
import os
import streamlit as st
from src.pipe.prediction_pipeline import PredictionPipeline

# Page Configuration
st.set_page_config(
    page_title="Smart Career Advisor AI",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: black;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 2rem auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stSelectbox > div {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #e0e7ff;
    }
    .stSelectbox > div:hover {
        background-color: #f8fafc;
        border-color: #2563eb;
    }
    .stProgress > div > div > div {
        background-color: #2563eb;
    }
    h1 {
        color: #1e293b;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        color: #334155;
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }
    h3 {
        color: #475569;
        font-size: 1.4rem;
        margin-bottom: 0.6rem;
    }
    .stMarkdown {
        color: #1e293b;
    }
    .suggestion-text {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2563eb;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f3ff 100%);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Introduction
st.title('🎯 Smart Career Advisor AI')
st.markdown("""
    <div style='background-color: #ffffff; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);'>
        <p style='color: #1e293b; font-size: 1.1rem;'>
            Welcome to your personalized career advisor! This AI-powered tool will help you discover 
            career paths that align with your skills, interests, and personality traits. 
            Please answer the following questions to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Progress Bar
progress_bar = st.progress(0)

# Questions Container
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('🎨 Professional Interests')
        interest = st.selectbox(
            'What field excites you the most?',
            ['art', 'technology', 'business', 'medicine', 'science'],
            help="Select the field that aligns with your professional interests"
        )
        progress_bar.progress(10)

        st.subheader('📚 Education Background')
        education = st.selectbox(
            'What is your highest level of education?',
            ['masters', 'phd', 'diploma', 'bachelors', 'high school'],
            help="Select your highest completed education level"
        )
        progress_bar.progress(30)

        st.subheader('💡 Problem Solving Approach')
        problem_solving = st.selectbox(
            'How do you typically approach problem-solving?',
            ['creative', 'logical', 'intuitive', 'practical', 'analytical'],
            help="Select your preferred problem-solving style"
        )
        progress_bar.progress(50)

        st.subheader('🤝 Communication Style')
        communication = st.selectbox(
            'How would you describe your communication skills?',
            ['excellent', 'poor', 'fluent', 'average', 'outstanding'],
            help="Select the level that best describes your communication abilities"
        )
        progress_bar.progress(70)

        st.subheader('🎭 Personality Traits')
        personality = st.selectbox(
            'Which personality trait best describes you?',
            ['thinker', 'extrovert', 'doer', 'leader', 'introvert'],
            help="Select your dominant personality trait"
        )
        progress_bar.progress(90)

    with col2:
        st.subheader('⚡ Skill Level')
        skill_level = st.selectbox(
            'What is your current professional skill level?',
            ['expert', 'advanced', 'intermediate', 'beginner'],
            help="Select your current professional expertise level"
        )
        progress_bar.progress(20)

        st.subheader('🏢 Work Environment')
        work_style = st.selectbox(
            'What type of work environment do you prefer?',
            ['flexible', 'team', 'remote', 'solo', 'office'],
            help="Select your preferred work environment"
        )
        progress_bar.progress(40)

        st.subheader('💻 Technical Proficiency')
        tech = st.selectbox(
            'How would you rate your technical skills?',
            ['very high', 'high', 'low', 'medium'],
            help="Select your level of technical expertise"
        )
        progress_bar.progress(60)

        st.subheader('👥 Leadership Style')
        leadership = st.selectbox(
            'How would you describe your leadership approach?',
            ['strong', 'moderate', 'collaborative', 'visionary'],
            help="Select your preferred leadership style"
        )
        progress_bar.progress(80)

        st.subheader('✨ Creativity Level')
        creativity = st.selectbox(
            'How would you rate your creative thinking abilities?',
            ['medium', 'very high', 'low', 'high', 'exceptional'],
            help="Select your level of creative thinking"
        )
        progress_bar.progress(100)

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline(
    Interest=interest,
    Skill_Level=skill_level,
    Work_Style=work_style,
    Education=education,
    Personality=personality,
    Problem_Solving=problem_solving,
    Tech=tech,
    Communication=communication,
    Leadership=leadership,
    Creativity=creativity
)

# Get Career Suggestion
if st.button('🔍 Get Career Suggestions'):
    with st.spinner('Analyzing your profile...'):
        prediction = prediction_pipeline.predict()[0]
        
        st.markdown("""
            <div class="result-box">
                <h3 style='color: #2563eb;'>🎯 Your Career Match</h3>
                <p style='color: #475569;'>Based on your responses, we recommend:</p>
                <div class="suggestion-text">{}</div>
                <p style='color: #475569; text-align: center;'>This career path aligns with your skills, interests, and personality traits.</p>
            </div>
            """.format(prediction), unsafe_allow_html=True)
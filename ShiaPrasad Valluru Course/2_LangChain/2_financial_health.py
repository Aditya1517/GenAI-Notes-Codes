from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def create_financial_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    financial_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","""You are an experienced financial advisor in India who specializes in personal finance management. Analyze the customer's financial situation and provide detailed insights. Present the output in a clean format with clear sections."""),
            ("human", """Perform initial financial analysis for the client in India:
                - Monthly Income: {monthly_income}
                - Monthly Expenses: {monthly_expenses}
                - Current Savings: {current_savings}

            Create a comprehensive assessment of their financial health.

                Following is the expected output
                A detailed financial analysis in markdown format including:
                - Current Financial Health Assessment
                - Cash Flow Analysis
                - Savings Potential
                - Risk Capacity Evaluation""",
            )
        ]
    )
    
    return financial_analysis_prompt | llm | StrOutputParser()


st.set_page_config(
    page_title = "Financial Health Analysis",
    page_icon = "💰",
    layout = "wide"
)
st.title("Personal Financial Health Analysis")

with st.sidebar:
    st.header("Input Your Financial Details")
    monthly_income = st.number_input("Monthly Income (INR)", min_value=0.0, value=150000.0)
    monthly_expenses = st.number_input("Monthly Expenses (INR)", min_value=0.0, value=80000.0)
    current_savings = st.number_input("Current Savings (INR)", min_value=0.0, value=500000.0)
    analyze_button = st.button("Analyze Financial Health")
    
    
if analyze_button:
    with st.spinner("Analyzing your financial health..."):
        chain = create_financial_chain()
        input = {
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "current_savings": current_savings
        }
        
        
        analysis_result = chain.invoke(input)
        
        st.markdown(analysis_result)
        
else:
    st.info("Please enter your financial details in the sidebar and click 'Analyze Financial Health' to get started.")
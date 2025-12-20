from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def create_financial_analysis_chain():

    financial_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced financial advisor in India who specializes in
            personal finance management. Analyze the customer's financial situation and provide
            detailed insights. Present the output in a clean format with clear sections."""),
        ("user", """Perform initial financial analysis for the client in india:
            - Monthly Income: {monthly_income}
            - Monthly Expenses: {monthly_expenses}
            - Current Savings: {current_savings}
        Create a comprehensive assessment of their financial health.

        Following is the expected output
            A detailed financial analysis in markdown format including:
                - Current Financial Health Assessment
                - Cash Flow Analysis
                - Savings Potential
                - Risk Capacity Evaluation""")
    ])
    
    return financial_analysis_prompt | llm | StrOutputParser()


def create_investment_recommendation_chain():
    investment_recommendation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced Indian financial advisor specializing in
            investment planning. Consider Indian investment options like Fixed Deposits,
            Mutual Funds, PPF, and NPS. Present recommendations in a clean format with clear sections."""),
        ("user", """Based on the financial analysis,
            - recommend investment strategies that can generate minimum 10 to 12% CAGR and maximum from 15
            - Match their risk tolerance: {risk_tolerance}
            - Align with their {investment_years} year timeline
            - Maximize potential returns while managing risk
            - you should concentrate on financial retirement in next 15 years. Consider inflation of 7% i
            - Consider that annual income increases by 10% per annum
            - Amount after expenses (considering inflation of 7% per annum) can be invested more annually.
            - You believe that if u get 4% per annum (after removing inflation) return on your investments
            - create investment plan so that customer can achieve financial independence in next 15 years

        Previous analysis: {financial_analysis}

        Below is the expected output:
            An investment strategy including:
            - Recommended Asset Allocation. Give how much amount to invest in each asset based on current
            - Specific Investment Vehicles with amounts exactly
            - Expected Returns based on current and future investments according plan
            - current monthly income, monthly expenses""")
    ])

    return investment_recommendation_prompt | llm | StrOutputParser()

chain = RunnablePassthrough() | {
            "financial_analysis":create_financial_analysis_chain(),
            "original_input":RunnablePassthrough()
        } | {
            "investment_recommendation":{
                "financial_analysis": lambda x: x["financial_analysis"],
                "risk_tolerance": lambda x: x["original_input"]["risk_tolerance"],
                "investment_years": lambda x: x["original_input"]["investment_years"]
            } | create_investment_recommendation_chain()
        }
        
        
inputs = {
    "monthly_income": 150000,
    "monthly_expenses": 80000,
    "current_savings": 500000,
    "risk_tolerance": "Moderate",
    "investment_years": 15
}


result = chain.invoke(inputs)

print(result)
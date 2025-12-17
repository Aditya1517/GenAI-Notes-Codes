from strands import Agent
from strands_tools import calculator, current_time # these are the tools


# create an agent with tools
agent = Agent(tools = [calculator,current_time])

# ask agents questions that uses available tools
# this uses default model claude 3.7
message="""
    My birth date is 15th November 2002, tell me my age in days
"""

agent(message)
from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import calculator, current_time

# create an ollama model instance
model_id = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:latest"
)

agent = Agent(mode=model_id, tools=[calculator, current_time])

message="""
    My birth date is 15th November 2002, tell me my age in days
"""

agent(message)
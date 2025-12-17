from strands import Agent
from strands.models import BedrockModel
from strands.tools import file_read, file_write   # FIXED import

# Step 1: Define the model
model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
)

# Step 2: Define an improved system prompt template
system_prompt = """
You are a helpful personal assistant with access to local file tools.

Your capabilities:
1. Read, understand, and summarize file contents.
2. Create and write text to new or existing files.
3. List directory contents and describe files.
4. Summarize or transform provided text.

When a user asks a question involving file access:
- Use the `file_read` tool to read files.
- Use the `file_write` tool to write or create files.
- If a file path is unclear, ask the user to clarify.

Always think step-by-step and explain the action before using a tool.

User query:
{input}
"""

# Step 3: Create the agent
agent = Agent(
    model=model,
    system_prompt=system_prompt,
    tools=[file_read, file_write],
)

# Step 4: Run the agent (recommended `.invoke` call)
response = agent.invoke(
    {"input": "What is the content of the file 'docs/chapter10.txt' and summarize it in under 100 words?"}
)

print(response)

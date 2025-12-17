from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient

# Connect to an MCP server using stdio transport
stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uv",
        args=["run", "main.py"]
    )
))

# Create an agent with MCP tools
with stdio_mcp_client:

    # Get the tools from the MCP server
    tools = stdio_mcp_client.list_tools_sync()

    # Create an agent with these tools
    agent = Agent(tools=tools)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        if not user_input:
            continue

        print("Agent:")
        response = agent(user_input)   # <-- Correct invocation
        print(response)

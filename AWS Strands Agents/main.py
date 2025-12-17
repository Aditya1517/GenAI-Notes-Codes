from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("My dice roll server.")

@mcp.tool()
def dice_roll(sides: int=6) -> str:
    """Roll a dice."""
    return f"You rolled a {random.randint(1, sides)}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
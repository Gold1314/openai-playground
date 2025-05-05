import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

model = ChatOpenAI(model="gpt-4", api_key=api_key)
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages":"Analyze how revenue of MSFT is changing over time."})
            print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())
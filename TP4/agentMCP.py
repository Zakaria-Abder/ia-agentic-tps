import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

async def main():
    try:
        client = MultiServerMCPClient(
            {
                "local_server": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["ressources/mcp_local_server.py"],
                    }
                }
        )

        tools = await client.get_tools()   

        resources = await client.get_resources("local_server")

        prompt = await client.get_prompt("local_server", "prompt")
        prompt = prompt[0].content

        model = ChatOllama(
        model="llama3.2:3b", 
        temperature=0
        )
        agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt
        )
        config = {"configurable": {"thread_id": "1"}}
        response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapt-ers library")]},

        config=config
        )
        print(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
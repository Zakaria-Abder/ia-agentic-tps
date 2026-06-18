from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, ToolMessage



# PARTIE 1 : CONTEXTE

@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

# PARTIE 2 : AGENT SANS CONTEXTE

model = ChatOllama(
model="llama3.2:3b", 
)
agent = create_agent(model=model,
context_schema=ColourContext)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)
print(response['messages'][-1].content)

# PARTIE 3 : OUTILS CONTEXTE


@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour


@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

# PARTIE 5 : CHANGEMENT DE CONTEXT 

response = agent.invoke(
     {"messages": [HumanMessage(content="What is my favourite colour?")]}, 
            context=ColourContext(favourite_colour="green") ) 

print(response['messages'][-1].content)

# PARTIE 5 : ETAT PERSONNALISE

class CustomState(AgentState):
    favourite_colour: str


# PARTIE 6 : MODIFIER ETAT

@tool
def update_favourite_colour(
    favourite_colour: str,
    runtime: ToolRuntime
) -> Command:
    """Update favourite colour in state"""

    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    content="Favourite colour updated",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


# PARTIE 7 : RECUPERATION ETAT

@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read favourite colour from state"""

    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"


# =====================================================
# TEST 1 : AGENT AVEC CONTEXTE
# =====================================================

print("\n===== TEST CONTEXTE =====")

agent_context = create_agent(
    model=model,
    tools=[
        get_favourite_colour,
        get_least_favourite_colour
    ],
    context_schema=ColourContext
)

response = agent_context.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is my favourite colour?"
            )
        ]
    },
    context=ColourContext()
)

print(response["messages"][-1].content)


# =====================================================
# TEST 2 : CHANGEMENT CONTEXTE
# =====================================================

print("\n===== CHANGEMENT CONTEXTE =====")

response = agent_context.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is my favourite colour?"
            )
        ]
    },
    context=ColourContext(
        favourite_colour="green"
    )
)

print(response["messages"][-1].content)


# =====================================================
# TEST 3 : ETAT MEMORISE
# =====================================================

print("\n===== ETAT AGENT =====")

agent_state = create_agent(
    model=model,
    tools=[
        update_favourite_colour,
        read_favourite_colour
    ],
    state_schema=CustomState,
    checkpointer=InMemorySaver()
)

thread = {
    "configurable": {
        "thread_id": "1"
    }
}

response = agent_state.invoke(
    {
        "messages": [
            HumanMessage(
                content="My favourite colour is green"
            )
        ]
    },
    thread
)

print(response["messages"][-1].content)

response = agent_state.invoke(
    {
        "messages": [
            HumanMessage(
                content="What's my favourite colour?"
            )
        ]
    },
    thread
)

print(response["messages"][-1].content)
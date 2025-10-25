from utils import load_prompt
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from typing import Optional
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

# Load the prompt
# Now you can easily load any prompt from the prompts folder
prompt_data = load_prompt("real_world_agent")  # .yaml extension is optional
system_prompt = prompt_data["system_prompt"]

# Define the tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


class Context(BaseModel):
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


model = init_chat_model(
    "openai:gpt-4o",
    temperature=0.1,
    timeout=10,
    max_tokens=1000
    )

class ResponseFormat(BaseModel):
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    
    # Any interesting information about the weather if available
    weather_conditions: Optional[str] = None

# Create a checkpointer
checkpointer = InMemorySaver()

# Create an agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather_for_location, get_user_location],
    system_prompt=system_prompt,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response["structured_response"])
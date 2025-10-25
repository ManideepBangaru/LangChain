# static model
from textwrap import wrap
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

agent = create_agent(
    model="openai:gpt-5",
    system_prompt="You are a helpful assistant",
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the meaning of life ?"}]})
print(response["messages"][-1].content)

print("--------------------------------")

# For more control over the model configuration, we can use the following:
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    timeout=30,
    max_tokens=1000
)

agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant",
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the meaning of life ?"}]})
print(response["messages"][-1].content)

print("--------------------------------")

# Dynamic model
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dotenv import load_dotenv
load_dotenv()

basic_model = ChatOpenAI(model="gpt-4o")
advanced_model = ChatOpenAI(model="gpt-5")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    # tools=tools,
    middleware=[dynamic_model_selection]
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the meaning of life ?"}]})
print(response["messages"][-1].content)

print("--------------------------------")

# Dynamic system prompt
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv
load_dotenv()

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="openai:gpt-4o",
    # tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
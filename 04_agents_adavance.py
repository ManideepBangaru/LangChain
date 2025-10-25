# Structured Output
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-5")

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

@tool
def get_contact_info(name: str) -> ContactInfo:
    """Get contact information for a given name."""
    return ContactInfo(name=name, email="test@example.com", phone="1234567890")

# agent = create_agent(
#     model=model,
#     tools=[get_contact_info],
#     system_prompt="You are a helpful assistant that can get contact information for a given name.",
#     response_format=ContactInfo
# )

# response = agent.invoke({"messages": [{"role": "user", "content": "What is the contact information for John Doe?"}]})
# print(response["messages"][-1].content)

print("--------------------------------")

# using providers strategy
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model=model,
    tools=[get_contact_info],
    system_prompt="You are a helpful assistant that can get contact information for a given name.",
    response_format=ProviderStrategy(ContactInfo)
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the contact information for John Doe?"}]})
print(response["messages"][-1].content)

print("--------------------------------")
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("openai:gpt-4o")

# response = model.invoke("Why do parrots talk?")
# print(response.content)

# print("--------------------------------")

# # Using messages format

# from langchain.messages import HumanMessage, AIMessage, SystemMessage

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."}
# ]

# response = model.invoke(conversation)
# print(response)

# print("--------------------------------")

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to French."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]

# response = model.invoke(conversation)
# print(response)  # AIMessage("J'adore créer des applications.")

# Structured outputs
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")
    producer: str = Field(...,description="The producer of the movie")

model_w_schema = model.with_structured_output(Movie, include_raw=True)

response = model_w_schema.invoke("What is the best movie of all time?")
print(response)
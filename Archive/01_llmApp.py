from dotenv import main
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from langserve import add_routes

# Loading openai api key
_ = main.load_dotenv(main.find_dotenv())

# Initiate and set model
model_name = 'gpt-4o-mini'
model = ChatOpenAI(model = model_name)

# create a prompt template
prompt_template = ChatPromptTemplate.from_template(""" Translate below text into {language} \
                                                        text : {text} """)

# Create a chain˛
chain = prompt_template | model | StrOutputParser()

# # test the chain
# response = chain.invoke({"language":"Telugu","text":"Hi, how are you"})
# print(response)

# create app definition
app = FastAPI(
    title="Langchian Server",
    version="1.0",
    description="A Simple API server using LangChain's Runnable interfaces"
)

# Adding chain routes
add_routes(
    app,
    chain,
    path='/chain'
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host = 'localhost', port=8000)
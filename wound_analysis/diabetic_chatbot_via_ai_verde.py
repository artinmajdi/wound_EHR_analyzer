import os

# We will add the api key as an environment variable.
# You can set this value directrly form the shell,
# instead of explicitly setting it on code.
os.environ["OPENAI_API_KEY"] = "sk-h8JtQkCCJUOy-TAdDxCLGw"
# We need a custom endpoint, as we will be calling Verde's LLM
API_ENDPOINT = "https://llm-api.cyverse.ai"


from langchain_openai import ChatOpenAI 		# We use the OpenAI protocol, but are using another provider (Verde)

# We will connect to Mistral Instruct v0.3 through Verde
# Notice how we need to specify the API endpoint
model = ChatOpenAI(model="Meta-Llama-3.1-70B-Instruct-quantized", base_url=API_ENDPOINT)

# Do a test call
from pprint import pprint
response = model.invoke("Hello, who are you?")

pprint(response)


# Let's build a conversation history

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a software developer, expert in the use of LLMs and langchain. Help the user answering the questions using didactic examples"),										# Use the system to establish the task of the LLM
    HumanMessage(content="What is Mistral and why would I need to use it"),		# This is the first "Human" message
]

# Instead of invoking with a string, pass the whole message history
response = model.invoke(messages)

# Add the response to the history
messages.append(response)

# Let's peek into the response
print(response.content)


messages.append(HumanMessage("Give me a short example of a python script that calls the aforementioned LLM"))

response = model.invoke(messages)
messages.append(response)

print(response.content)

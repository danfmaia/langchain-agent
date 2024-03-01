from langchain.llms import OpenAI

# Ensure your OPENAI_API_KEY environment variable is set
openai_llm = OpenAI()

# Test by generating a simple text
response = openai_llm.generate("Hello, world!")
print(response)

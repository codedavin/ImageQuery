# query_handler.py
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

def explain_and_answer(caption: str, query: str) -> str:
    prompt_template = (
        "You are a helpful assistant who explains image captions in great detail and answers queries based on them. "
        "Given the image caption below and a user query, provide a detailed explanation that connects the two.\n\n"
        "Image Caption: {caption}\n"
        "User Query: {query}\n\n"
        "Explanation and Answer:"
    )
    prompt = PromptTemplate(input_variables=["caption", "query"], template=prompt_template)
    llm = Ollama(model="llama3.1")
    chain = LLMChain(llm=llm, prompt=prompt)
    explanation = chain.run(caption=caption, query=query)
    return explanation
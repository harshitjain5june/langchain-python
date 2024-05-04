from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()
output_parser = StrOutputParser()
# messages = [SystemMessage(content="You are a complete wikipedia who knows everything")]
# messages.append(HumanMessage(content="how to make a sandwich"))
# res = llm.invoke(messages)
llm = AzureChatOpenAI(deployment_name="langchain", model_name="gpt-35-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm | output_parser

res=chain.invoke({"input": "How to download movie using torrent"})
print(res)
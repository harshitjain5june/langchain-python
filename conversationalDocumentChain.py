from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from BaseClass import setUpRetriever

langClient = setUpRetriever("https://docs.smith.langchain.com/user_guide")
# First we need a prompt that we can pass into an LLM to generate this search query
prompt_to_generate_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

#Final prompt after getting the search query from above prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(langClient.llm, prompt)

retriever_chain = create_history_aware_retriever(langClient.llm, langClient.retriever, prompt_to_generate_search_query)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

#Just to check what it returns
# retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })


retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response["answer"])
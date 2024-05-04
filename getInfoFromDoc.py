from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from BaseClass import setUpRetriever


langClient =  setUpRetriever("https://python.langchain.com/docs/get_started/quickstart/")
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(langClient.llm, prompt)

retrieval_chain = create_retrieval_chain(langClient.retriever, document_chain)

response = retrieval_chain.invoke({"input":"Explain in detail the process of Agent?"})

print(response["answer"])



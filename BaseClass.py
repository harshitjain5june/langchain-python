from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



class setUpRetriever:

    def __init__(self, docUrl) -> None:
        load_dotenv()
        self.llm = AzureChatOpenAI(deployment_name="langchain", model_name="gpt-35-turbo")
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment="embeddings", model="text-embedding-ada-002")
        self.loader = WebBaseLoader(docUrl)
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.documents = self.text_splitter.split_documents(self.docs)
        self.vector = FAISS.from_documents(self.documents, self.embeddings)
        self.retriever = self.vector.as_retriever()
        


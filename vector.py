from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db1"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents=[]
    ids=[]
    
    for i , row in df.iterrows():
        document = Document(
            page_content = row["Title"] + "" + row["Review"],
            metadata = {
                "Rating": row["Rating"],
                "Date": row["Date"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restuarant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents,ids= ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


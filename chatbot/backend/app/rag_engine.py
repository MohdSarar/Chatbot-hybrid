
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

class LLMEngine:
    def init(self, df_formations: pd.DataFrame):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.vector_store = None
        self.qa_chain = None
        self.initialize_rag(df_formations)

    def _df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """Convertit le DataFrame des formations en documents LangChain"""
        docs = []
        for _, row in df.iterrows():
            content = f"""
            Formation: {row['titre']}
            Objectifs: {', '.join(row['objectifs'])}
            Prérequis: {', '.join(row['prerequis'])}
            Programme: {', '.join(row['programme'])}
            Public: {', '.join(row['public'])}
            Lien: {row['lien']}
            """
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "formations",
                    "titre": row["titre"],
                    "type": "formation"
                }
            ))
        return docs

    def initialize_rag(self, df_formations: pd.DataFrame):
        """Initialise le système RAG avec les données des formations"""
        # Conversion du DataFrame en documents
        documents = self._df_to_documents(df_formations)
        
        # Découpage des textes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Création du vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Création de la chaîne de conversation
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def generate_response(self, question: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Génère une réponse contextuelle basée sur l'historique et les données RAG"""
        if not self.qa_chain:
            return {"answer": "Le système RAG n'est pas initialisé", "sources": []}
        
        # Conversion de l'historique au format LangChain
        converted_history = []
        for msg in chat_history:
            converted_history.append((msg['content'], msg['reply']))
        
        # Appel au modèle
        result = self.qa_chain({
            "question": question,
            "chat_history": converted_history
        })
        
        # Formatage des sources
        sources = list(set(
            f"{doc.metadata['titre']}" 
            for doc in result['source_documents']
        ))
        
        return {
            "answer": result['answer'],
            "sources": sources
        }
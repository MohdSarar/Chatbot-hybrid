# memory.py
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory, CombinedMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import load_summarize_chain
from app.schemas import SessionState

def init_memory(session: SessionState) -> CombinedMemory:
    """Initialise ou récupère la mémoire persistante liée à la session."""
    if not hasattr(session, "memory"):
        buffer_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=False,
            human_prefix="Utilisateur",
            ai_prefix="Assistant"
        )
        entity_memory = ConversationEntityMemory(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            memory_key="entities",
            input_key="question"
        )
        session.memory = CombinedMemory(memories=[buffer_memory, entity_memory])
    return session.memory

def summarize_history(history: str) -> str:
    """Résume l'historique si trop long avec GPT-3.5 Turbo."""
    if len(history) < 1000:
        return history
    
    summarizer = load_summarize_chain(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
        chain_type="map_reduce"
    )
    return summarizer.run(history)
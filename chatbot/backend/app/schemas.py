from pydantic import BaseModel, EmailStr, validator, Field
from typing import List, Optional
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory

# --------------------------------------------------
# Modèles Pydantic adaptés au frontend
# --------------------------------------------------
class UserProfile(BaseModel):
    name: str = Field(..., description="User's name")
    email: Optional[EmailStr] = Field(None, description="User's email address")
    objective: str = Field(..., description="User's career objective")
    level: str = Field(..., description="User's current level")
    knowledge: str = Field(default="", description="User's knowledge/skills - can be empty")  # ✅ FIXED: Can be empty
    pdf_content: Optional[str] = Field(None, description="Extracted PDF content")
    recommended_course: Optional[str] = Field(None, description="Recommended course")
    
    @validator('name', 'objective', 'level')  # ✅ REMOVED knowledge from required validation
    def validate_required_strings(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('knowledge')  # ✅ SEPARATE validator for knowledge - allows empty
    def validate_knowledge(cls, v):
        # Allow empty knowledge, just clean it up
        return v.strip() if v else ""
    
    @validator('email')
    def validate_email_field(cls, v):
        # Convert empty string to None for email field
        return v if v and v.strip() else None
    
    @validator('pdf_content')
    def validate_pdf_content(cls, v):
        # Convert empty string to None
        return v if v and v.strip() else None
    
    @validator('level')
    def normalize_level(cls, v):
        # Normalize level values to match frontend
        level_mapping = {
            'débutant': 'Débutant',
            'intermédiaire': 'Intermédiaire', 
            'avancé': 'Avancé',
            'expert': 'Expert'
        }
        return level_mapping.get(v.lower(), v)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Jean Dupont",
                "email": "jean.dupont@email.com",
                "objective": "Devenir data analyst",
                "level": "Débutant",
                "knowledge": ""  # ✅ Can be empty
            }
        }

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        # Allow any content, just ensure it's a string
        return str(v) if v is not None else ""

class SendEmailRequest(BaseModel):
    profile: UserProfile
    chatHistory: List[ChatMessage] = Field(default_factory=list)

class RecommendRequest(BaseModel):
    profile: UserProfile
    
    class Config:
        schema_extra = {
            "example": {
                "profile": {
                    "name": "Jean Dupont",
                    "email": "jean.dupont@email.com",
                    "objective": "Devenir data analyst",
                    "level": "Débutant",
                    "knowledge": ""  # ✅ Can be empty
                }
            }
        }

class RecommendResponse(BaseModel):
    recommended_course: str
    reply: str
    details: Optional[dict] = None

class QueryRequest(BaseModel):
    profile: UserProfile
    history: List[ChatMessage] = Field(default_factory=list)
    question: str = Field(..., description="User's question")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not str(v).strip():
            raise ValueError('Question cannot be empty')
        return str(v).strip()
    
    class Config:
        schema_extra = {
            "example": {
                "profile": {
                    "name": "Jean Dupont",
                    "email": "jean.dupont@email.com",
                    "objective": "Devenir data analyst",
                    "level": "Débutant",
                    "knowledge": ""  # ✅ Can be empty
                },
                "history": [
                    {
                        "role": "user",
                        "content": "Bonjour"
                    },
                    {
                        "role": "assistant", 
                        "content": "Bonjour Jean ! Comment puis-je vous aider ?"
                    }
                ],
                "question": "Quelles formations recommandez-vous ?"
            }
        }

class QueryResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    next_action: Optional[str] = None
    recommended_course: Optional[dict] = None

class SessionState(BaseModel):
    user_id: str
    current_title: Optional[str] = None
    last_intent: Optional[str] = None
    recommended_course: Optional[str] = None
    buffer_memory: Optional[ConversationBufferMemory] = None
    entity_memory: Optional[ConversationEntityMemory] = None
    
    class Config:
        arbitrary_types_allowed = True

class SearchFilters(BaseModel):
    certifiant: Optional[bool] = None
    modalite: Optional[str] = None
    lieu: Optional[str] = None
    duree_max: Optional[int] = None
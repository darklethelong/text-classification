"""
Conversation Schemas
-------------------
This module contains schema models for conversations.
"""

from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    """Model for conversation analysis request.
    
    Attributes:
        text: Full conversation text between agent and caller
    """
    text: str = Field(
        ..., 
        title="Conversation text", 
        description="Full conversation text between agent and caller",
        example="""Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.
Caller: This is ridiculous! I'm paying for a service I'm not receiving.
Agent: I understand your frustration. Let me schedule a technician visit with our highest priority.
Caller: I want a refund for the days I haven't had service.
Agent: That's a reasonable request. I'll process a credit for the days affected."""
    )


class ConversationResponse(BaseModel):
    """Model for conversation analysis response.
    
    Attributes:
        is_complaint: Whether the conversation contains a complaint
        confidence: Confidence score for the prediction
        processed_at: Timestamp of when the prediction was made
    """
    is_complaint: bool = Field(
        ...,
        title="Is Complaint",
        description="Whether the conversation contains a complaint"
    )
    confidence: float = Field(
        ...,
        title="Confidence",
        description="Confidence score for the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    processed_at: str = Field(
        ...,
        title="Processed At",
        description="Timestamp of when the prediction was made"
    ) 
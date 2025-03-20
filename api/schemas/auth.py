"""
Authentication Schemas
---------------------
This module contains schema models for authentication.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    """Model for authentication token response.
    
    Attributes:
        access_token: JWT access token
        token_type: Token type (e.g., "bearer")
    """
    access_token: str = Field(
        ...,
        title="Access Token",
        description="JWT access token"
    )
    token_type: str = Field(
        ...,
        title="Token Type",
        description="Type of token (e.g., 'bearer')"
    )


class TokenData(BaseModel):
    """Model for token data.
    
    Attributes:
        username: Username associated with the token
    """
    username: Optional[str] = Field(
        None,
        title="Username",
        description="Username associated with the token"
    )


class UserBase(BaseModel):
    """Base model for user data.
    
    Attributes:
        username: Unique username
        email: Email address
        full_name: Full name of the user
        disabled: Whether the user is disabled
    """
    username: str = Field(
        ...,
        title="Username",
        description="Unique username"
    )
    email: Optional[str] = Field(
        None,
        title="Email",
        description="Email address"
    )
    full_name: Optional[str] = Field(
        None,
        title="Full Name",
        description="Full name of the user"
    )
    disabled: Optional[bool] = Field(
        None,
        title="Disabled",
        description="Whether the user is disabled"
    )


class UserInDB(UserBase):
    """Model for user data stored in the database.
    
    Extends the UserBase model with password hash.
    
    Attributes:
        hashed_password: Hashed password
    """
    hashed_password: str = Field(
        ...,
        title="Hashed Password",
        description="Hashed password"
    )


class UserCreate(UserBase):
    """Model for creating a new user.
    
    Extends the UserBase model with password.
    
    Attributes:
        password: Plain text password (will be hashed before storing)
    """
    password: str = Field(
        ...,
        title="Password",
        description="User password",
        min_length=8
    )


class User(UserBase):
    """Model for user data returned to clients.
    
    This is the model used for user data in responses.
    """
    class Config:
        """Pydantic configuration for the User model."""
        orm_mode = True 
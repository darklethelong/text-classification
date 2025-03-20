"""
Authentication Router
-------------------
This module contains routes for authentication.
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from config import settings
from schemas.auth import Token, User
from services.auth import (USERS_DB, authenticate_user, create_access_token,
                         get_current_active_user)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get an OAuth2 access token.
    
    This endpoint is used to authenticate a user and get a JWT token that can be used for
    authenticated API calls.
    """
    user = authenticate_user(USERS_DB, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User {form_data.username} logged in successfully")
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    
    This endpoint returns information about the currently authenticated user.
    """
    return current_user 
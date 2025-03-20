"""
API Tests
---------
This module contains tests for the API endpoints.
"""

import unittest
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """Create a mock for the model service."""
    with patch("routers.complaints.model_service") as mock:
        # Set up mock responses
        mock.model = Mock()
        mock.vocab = Mock()
        mock.vocab.word2idx = {"hello": 1, "world": 2}
        mock.predict.return_value = (1, 0.75)
        yield mock


@pytest.fixture
def auth_token(client):
    """Get an authentication token for testing."""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "adminpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return response.json()["access_token"]


def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_token_endpoint_valid_credentials(client):
    """Test the token endpoint with valid credentials."""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "adminpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_token_endpoint_invalid_credentials(client):
    """Test the token endpoint with invalid credentials."""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "wrongpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 401


def test_predict_endpoint_authenticated(client, auth_token, mock_model_service):
    """Test the predict endpoint with authentication."""
    conversation = "Caller: Hello\nAgent: World"
    response = client.post(
        "/api/predict",
        json={"text": conversation},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert "is_complaint" in response.json()
    assert "confidence" in response.json()
    assert "processed_at" in response.json()
    assert response.json()["is_complaint"] is True
    assert response.json()["confidence"] == 0.75


def test_predict_endpoint_unauthenticated(client):
    """Test the predict endpoint without authentication."""
    conversation = "Caller: Hello\nAgent: World"
    response = client.post(
        "/api/predict",
        json={"text": conversation}
    )
    assert response.status_code == 401


def test_user_info_authenticated(client, auth_token):
    """Test getting user info with authentication."""
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert "username" in response.json()
    assert response.json()["username"] == "admin"


def test_user_info_unauthenticated(client):
    """Test getting user info without authentication."""
    response = client.get("/users/me")
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"]) 
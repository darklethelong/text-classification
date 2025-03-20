"""
Stress Test Script
-----------------
This script uses Locust to perform stress testing on the API.

Run with:
    locust -f stress_test.py
"""

import json
import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests
from locust import HttpUser, between, task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Example conversations for testing
EXAMPLE_CONVERSATIONS = [
    # Complaint conversations
    """Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.
Caller: This is ridiculous! I'm paying for a service I'm not receiving.
Agent: I understand your frustration. Let me schedule a technician visit with our highest priority.
Caller: I want a refund for the days I haven't had service.
Agent: That's a reasonable request. I'll process a credit for the days affected.""",
    
    """Caller: I ordered a laptop from your website two weeks ago and it still hasn't arrived.
Agent: I apologize for the delay. Let me check on your order status.
Caller: The confirmation said it would arrive within 3-5 business days.
Agent: You're right, and I see here that it should have been delivered by now.
Caller: This is unacceptable. I need the laptop for work.
Agent: I completely understand. I'll expedite your order right away.
Caller: I want some compensation for this delay.
Agent: I'll certainly apply a discount to your order for the inconvenience.""",
    
    # Non-complaint conversations
    """Caller: Hi, I'm calling to upgrade my plan.
Agent: Hello! I'd be happy to help you with that. What plan are you interested in?
Caller: I saw the premium package online. It has more channels.
Agent: The premium package is a great choice. It includes 200+ channels, including HBO and Showtime.
Caller: That sounds good. How much would it cost?
Agent: The premium package is $89.99 per month. I can also offer a 3-month discount at $69.99 if you upgrade today.
Caller: That sounds like a good deal. Let's go with that.
Agent: Excellent! I'll process that upgrade right away for you.""",
    
    """Caller: Hello, I'd like to know more about your fiber internet options.
Agent: Hi there! I'd be glad to tell you about our fiber options. We offer speeds ranging from 100Mbps to 1Gbps.
Caller: The 1Gbps option sounds interesting. Is it available in my area?
Agent: I'd be happy to check that for you. Could you provide your address?
Caller: It's 123 Main Street, Anytown.
Agent: Thank you. Yes, I see that fiber is available at your location!
Caller: Great! And what's the monthly cost for the 1Gbps plan?
Agent: The 1Gbps plan is $79.99 per month with a 12-month agreement. We're also offering free installation right now.
Caller: Sounds good. I'd like to schedule an installation.
Agent: Wonderful! Let's set that up for you."""
]


class ComplaintDetectionUser(HttpUser):
    """
    Locust user class for testing the complaint detection API.
    
    This class simulates users making requests to the API with different patterns:
    - Authentication to get token
    - Checking the health endpoint
    - Predicting complaints from sample conversations
    """
    wait_time = between(1, 5)  # Wait between 1 and 5 seconds between tasks
    token = None
    
    def on_start(self):
        """
        Initialize the user session.
        
        This authenticates with the API to get a token that will be used
        for subsequent requests.
        """
        response = self.client.post(
            "/token",
            data={"username": "admin", "password": "adminpassword"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            logger.info("Authentication successful")
        else:
            logger.error(f"Authentication failed: {response.text}")
    
    @task(1)
    def health_check(self):
        """
        Check the health endpoint.
        
        This endpoint doesn't require authentication and is used to verify
        the API is running properly.
        """
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.text}")
    
    @task(3)
    def predict_complaint(self):
        """
        Make a prediction request to the API.
        
        This task picks a random conversation from the examples and sends it
        to the prediction endpoint.
        """
        if not self.token:
            logger.error("No token available, skipping prediction")
            return
            
        # Pick a random conversation
        conversation = random.choice(EXAMPLE_CONVERSATIONS)
        
        # Log request start time
        start_time = time.time()
        
        # Make the request
        with self.client.post(
            "/api/predict",
            json={"text": conversation},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            catch_response=True
        ) as response:
            # Calculate response time
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Prediction result: is_complaint={result['is_complaint']}, "
                    f"confidence={result['confidence']:.4f}, "
                    f"time={response_time:.4f}s"
                )
                response.success()
            else:
                logger.error(f"Prediction failed: {response.text}")
                response.failure(f"Prediction failed: {response.text}")
    
    @task(1)
    def get_user_info(self):
        """
        Get current user information.
        
        This task makes a request to get information about the authenticated user.
        """
        if not self.token:
            logger.error("No token available, skipping user info request")
            return
            
        with self.client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {self.token}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get user info failed: {response.text}")


if __name__ == "__main__":
    # This block allows running the script directly for testing
    # without using the Locust command-line interface
    
    # URL of the API
    base_url = "http://localhost:8000"
    
    # Authenticate
    response = requests.post(
        f"{base_url}/token",
        data={"username": "admin", "password": "adminpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"Authentication successful: {token}")
        
        # Health check
        response = requests.get(f"{base_url}/api/health")
        print(f"Health check: {response.json()}")
        
        # Make a prediction
        conversation = EXAMPLE_CONVERSATIONS[0]  # Use the first example
        response = requests.post(
            f"{base_url}/api/predict",
            json={"text": conversation},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction result: {result}")
        else:
            print(f"Prediction failed: {response.text}")
        
        # Get user info
        response = requests.get(
            f"{base_url}/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            print(f"User info: {response.json()}")
        else:
            print(f"Get user info failed: {response.text}")
            
    else:
        print(f"Authentication failed: {response.text}") 
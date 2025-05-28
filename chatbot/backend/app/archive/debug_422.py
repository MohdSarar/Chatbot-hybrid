#!/usr/bin/env python3
"""
Test script to debug 422 errors in the chatbot API
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_recommend_endpoint():
    """Test the /recommend endpoint"""
    print("\n=== Testing /recommend endpoint ===")
    
    # Valid request payload
    payload = {
        "profile": {
            "name": "Jean Dupont",
            "email": "jean.dupont@email.com",
            "objective": "Devenir data analyst",
            "level": "débutant",
            "knowledge": "Excel, SQL de base"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 422:
            print("❌ Validation Error - Check the request format")
            print("Expected format:")
            print(json.dumps(payload, indent=2))
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_query_endpoint():
    """Test the /query endpoint"""
    print("\n=== Testing /query endpoint ===")
    
    # Valid request payload
    payload = {
        "profile": {
            "name": "Jean Dupont",
            "email": "jean.dupont@email.com", 
            "objective": "Devenir data analyst",
            "level": "débutant",
            "knowledge": "Excel, SQL de base"
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
        "question": "Quelles formations recommandez-vous pour un débutant ?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 422:
            print("❌ Validation Error - Check the request format")
            print("Expected format:")
            print(json.dumps(payload, indent=2))
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_malformed_requests():
    """Test various malformed requests to understand validation"""
    print("\n=== Testing malformed requests ===")
    
    # Missing required fields
    test_cases = [
        {
            "name": "Missing profile",
            "payload": {"question": "Hello"}
        },
        {
            "name": "Missing name in profile",
            "payload": {
                "profile": {
                    "objective": "Test",
                    "level": "débutant",
                    "knowledge": "Excel"
                },
                "question": "Hello"
            }
        },
        {
            "name": "Empty string fields",
            "payload": {
                "profile": {
                    "name": "",
                    "objective": "Test",
                    "level": "débutant", 
                    "knowledge": "Excel"
                },
                "question": "Hello"
            }
        },
        {
            "name": "Wrong field types",
            "payload": {
                "profile": {
                    "name": 123,  # Should be string
                    "objective": "Test",
                    "level": "débutant",
                    "knowledge": "Excel"
                },
                "question": "Hello"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json=test_case['payload'],
                headers={"Content-Type": "application/json"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 422:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("🔍 Testing Chatbot API Endpoints...")
    print("Make sure your server is running on http://127.0.0.1:8000")
    
    # Test valid requests
    test_recommend_endpoint()
    test_query_endpoint()
    
    # Test malformed requests
    test_malformed_requests()
    
    print("\n✅ Testing completed!")
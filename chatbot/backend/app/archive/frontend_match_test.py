#!/usr/bin/env python3
"""
Test script that matches exactly what the frontend sends
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_frontend_exact_format():
    """Test with the exact format the frontend sends"""
    print("\n=== Testing Frontend Exact Format ===")
    
    # This matches exactly what your Angular frontend sends
    payload = {
        "profile": {
            "name": "Jean Dupont",
            "email": "",  # Frontend sends empty string, not null
            "objective": "Devenir data analyst",
            "level": "Débutant",  # Frontend uses capitalized form
            "knowledge": "",  # ✅ Frontend allows this to be empty
            "pdf_content": ""  # Frontend sends empty string
        }
    }
    
    print("Sending payload matching frontend:")
    print(json.dumps(payload, indent=2))
    
    try:
        # Test /recommend endpoint
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n📤 /recommend Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS - Backend now accepts empty knowledge!")
            result = response.json()
            print(f"Recommended: {result.get('recommended_course', 'N/A')}")
        else:
            print(f"❌ FAILED - Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_query_with_empty_knowledge():
    """Test /query endpoint with empty knowledge"""
    print("\n=== Testing /query with Empty Knowledge ===")
    
    payload = {
        "profile": {
            "name": "Marie Martin",
            "email": "",
            "objective": "Apprendre l'IA",
            "level": "Débutant",
            "knowledge": "",  # ✅ Empty knowledge
            "pdf_content": ""
        },
        "history": [],
        "question": "Bonjour, pouvez-vous m'aider ?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📤 /query Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS - Query works with empty knowledge!")
            result = response.json()
            print(f"Reply: {result.get('reply', 'N/A')[:100]}...")
        else:
            print(f"❌ FAILED - Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_various_knowledge_formats():
    """Test different knowledge field formats"""
    print("\n=== Testing Various Knowledge Formats ===")
    
    test_cases = [
        {"knowledge": "", "description": "Empty string"},
        {"knowledge": "   ", "description": "Whitespace only"},
        {"knowledge": "Excel", "description": "Single skill"},
        {"knowledge": "Excel, Python, SQL", "description": "Multiple skills"},
        {"knowledge": None, "description": "Null value"}  # This might fail, but let's test
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['description']}")
        
        payload = {
            "profile": {
                "name": f"Test User {i+1}",
                "email": "",
                "objective": "Test objective",
                "level": "Débutant",
                "knowledge": case['knowledge']
            }
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"  ✅ PASS - {case['description']}")
            else:
                print(f"  ❌ FAIL - {case['description']}: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ ERROR - {case['description']}: {e}")

if __name__ == "__main__":
    print("🔄 Testing Backend Compatibility with Frontend...")
    print("Make sure your server is running on http://127.0.0.1:8000")
    print("This test uses the EXACT format your Angular frontend sends")
    
    test_frontend_exact_format()
    test_query_with_empty_knowledge()
    test_various_knowledge_formats()
    
    print("\n✅ Testing completed!")
    print("If all tests pass, your backend now accepts frontend requests correctly!")
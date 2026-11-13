#!/usr/bin/env python3
"""Test script to check the server health endpoint."""

from fastapi.testclient import TestClient
from clipforge.app import create_app

app = create_app()
client = TestClient(app)

# Test the health endpoint
response = client.get("/api/system/health")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Test the root endpoint
response = client.get("/")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")
#!/usr/bin/env python3
"""
Script de test simple pour diagnostiquer les problèmes backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Test Backend")

# CORS simple
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Backend test OK", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/auth/test")
async def test_auth():
    return {"message": "Auth endpoint accessible", "status": "OK"}

@app.post("/api/v1/auth/login")
async def test_login(data: dict):
    return {
        "access_token": "test-token",
        "token_type": "bearer",
        "message": "Test login successful"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
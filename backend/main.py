import os
import httpx
from fastapi import FastAPI, Request, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pyzeebe import ZeebeClient, create_insecure_channel, ZeebeWorker
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import jwt
import uuid
from typing import Dict

load_dotenv()

# Zeebe Configuration (Defaults to local docker-compose-full with Keycloak)
ZEEBE_ADDRESS = os.getenv("ZEEBE_ADDRESS", "localhost:26500")

# Authentication Configuration (for Keycloak / Camunda Cloud)
ZEEBE_CLIENT_ID = os.getenv("ZEEBE_CLIENT_ID")
ZEEBE_CLIENT_SECRET = os.getenv("ZEEBE_CLIENT_SECRET")
ZEEBE_AUTHORIZATION_SERVER_URL = os.getenv("ZEEBE_AUTHORIZATION_SERVER_URL")
ZEEBE_TOKEN_AUDIENCE = os.getenv("ZEEBE_TOKEN_AUDIENCE", "zeebe-api")

# System Prompt Configuration
VOICE_AI_SYSTEM_PROMPT = os.getenv("VOICE_AI_SYSTEM_PROMPT", "You are a helpful voice assistant.")

# Globals and Models
zeebe_client = None
active_connections: Dict[str, WebSocket] = {}
HF_BASE_URL = "https://huggingface.co"

import grpc

class Message(BaseModel):
    text: str
    metadata: dict = {}

async def get_access_token():
    """Fetch OAuth2 token from the authorization server."""
    if not all([ZEEBE_CLIENT_ID, ZEEBE_CLIENT_SECRET, ZEEBE_AUTHORIZATION_SERVER_URL]):
        print(f"Skipping OAuth token fetch (missing credentials)")
        return None
    
    # Strip quotes if they exist in the env variables
    client_id = ZEEBE_CLIENT_ID.strip("'").strip('"') if ZEEBE_CLIENT_ID else ""
    client_secret = ZEEBE_CLIENT_SECRET.strip("'").strip('"') if ZEEBE_CLIENT_SECRET else ""
    auth_url = ZEEBE_AUTHORIZATION_SERVER_URL.strip("'").strip('"') if ZEEBE_AUTHORIZATION_SERVER_URL else ""
    audience = (ZEEBE_TOKEN_AUDIENCE or "zeebe-api").strip("'").strip('"')

    print(f"Fetching access token for client '{client_id}' from {auth_url}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                auth_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "audience": audience,
                },
                timeout=10.0
            )
            print(f"Token response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            if token:
                print("Successfully fetched access token.")
                # Decode without verification for debugging purposes
                try:
                    decoded = jwt.decode(token, options={"verify_signature": False})
                    print("--- TOKEN DEBUG ---")
                    print(f"Issuer (iss):   {decoded.get('iss')}")
                    print(f"Audience (aud): {decoded.get('aud')}")
                    print(f"Client ID:      {decoded.get('azp') or decoded.get('client_id')}")
                    print("-------------------")
                except Exception as e:
                    print(f"[Debug] Could not decode token: {e}")
            else:
                print(f"Token missing in response: {data}")
            return token
        except Exception as e:
            print(f"Failed to fetch OAuth token: {type(e).__name__}: {e}")
            return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global zeebe_client
    # Initialize Zeebe Client inside the event loop
    try:
        token = await get_access_token()
        target_address = ZEEBE_ADDRESS.strip("'").strip('"')
        
        if token:
            print(f"Connecting to Zeebe at {target_address} using secure composite credentials...")
            
            if "localhost" in target_address or "127.0.0.1" in target_address:
                # Local dev usually uses insecure_channel (composite credentials require SSL)
                channel_with_auth = grpc.aio.insecure_channel(target_address)
            else:
                # Production SaaS or Cloud using SSL + OAuth2 Bearer Token
                # This is the gold standard for gRPC auth with Camunda Cloud
                ssl_credentials = grpc.ssl_channel_credentials()
                
                # Use access_token_call_credentials for standard Bearer token injection
                auth_credentials = grpc.access_token_call_credentials(token)
                
                # Combine SSL/TLS with the Bearer token credentials
                composite_credentials = grpc.composite_channel_credentials(
                    ssl_credentials, 
                    auth_credentials
                )
                
                channel_with_auth = grpc.aio.secure_channel(
                    target_address, 
                    credentials=composite_credentials
                )
            
            zeebe_client = ZeebeClient(channel_with_auth)
            
            # Register a Job Worker to provide responses back from the Camunda process
            # This worker handles tasks of type 'provide-voice-response'
            worker = ZeebeWorker(channel_with_auth)

            @worker.task(task_type="provide-voice-response")
            async def handle_voice_response(botReply: str, uniqueId: str):
                # Clean the ID in case of string serialization artifacts
                clean_id = str(uniqueId).strip('"').strip("'")
                print(f"[Zeebe-Worker] Received botReply for session {clean_id}: {botReply}")
                
                if clean_id in active_connections:
                    ws = active_connections[clean_id]
                    try:
                        await ws.send_json({
                            "type": "botReply",
                            "content": botReply
                        })
                        print(f"[Zeebe-Worker] Relayed to frontend via WebSocket for session {clean_id}.")
                    except Exception as e:
                        print(f"[Zeebe-Worker] Failed to send via WS: {e}")
                else:
                    print(f"[Zeebe-Worker] No active WS session found for {clean_id}. Available: {list(active_connections.keys())}")
                return {"relayed": True}
            
            # Start the worker task in the background
            asyncio.create_task(worker.work())
            print(f"Registered Zeebe Job Worker for task type 'provide-voice-response'.")

        else:
            print(f"Connecting to Zeebe at {target_address} (plaintext)...")
            channel = create_insecure_channel(grpc_address=target_address)
            zeebe_client = ZeebeClient(channel)
            
    except Exception as e:
        print(f"Failed to connect to Zeebe: {e}")
    yield
    # Shutdown logic if any
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow the Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration API
@app.get("/api/config")
async def get_config():
    """Return backend configuration to the frontend."""
    return {
        "systemPrompt": VOICE_AI_SYSTEM_PROMPT
    }

# WebSocket for providing real-time bot replies from Zeebe Job Worker
@app.websocket("/ws/{unique_id}")
async def websocket_endpoint(websocket: WebSocket, unique_id: str):
    await websocket.accept()
    active_connections[unique_id] = websocket
    print(f"[WS] Session {unique_id} connected.")
    try:
        while True:
            # Keep the connection open - monitor for breaks
            data = await websocket.receive_text()
            # Respond with pong to keep alive if needed or echo for debug
            print(f"[WS] Received from {unique_id}: {data}")
    except WebSocketDisconnect:
        print(f"[WS] Session {unique_id} disconnected.")
        if unique_id in active_connections:
            del active_connections[unique_id]
    except Exception as e:
        print(f"[WS] Error on {unique_id}: {e}")
        if unique_id in active_connections:
            del active_connections[unique_id]

# 1. Proxy for Hugging Face files (Models and Voices)
@app.get("/hf-proxy/{path:path}")
async def proxy_hf(path: str):
    # This relays requests to huggingface.co/ (e.g. resolve/main/...)
    target_url = f"{HF_BASE_URL}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # We follow redirects because HF resolve URLs usually redirect to LFS storage
            # Increased timeout for large model files
            response = await client.get(target_url, follow_redirects=True, timeout=120.0)
            
            # Return the response exactly as HF returned it (including 404 for optional config files)
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/octet-stream")
            )
        except Exception as e:
            print(f"Proxy error for {target_url}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 2. API to relay transcribed text to Camunda
@app.post("/api/message")
async def send_to_camunda(msg: Message):
    # 'session_id' is the uniqueId passed from the frontend that stays same for the session
    session_id = msg.metadata.get("uniqueId", "default")
    # 'message_key' is a random UUID generated for each individual message to fulfill 
    # the requirement that the correlate/message key must be unique/random for each event.
    message_key = str(uuid.uuid4())
    
    print(f"[/api/message] Processing text: '{msg.text[:50]}{'...' if len(msg.text) > 50 else ''}' from {msg.metadata.get('source', 'unknown')} | session_id: {session_id} | message_key: {message_key}")
    
    try:
        # User defined message parameters:
        # - Message name: MSG_VOICE
        # - Key for text: userInput
        # - Correlation ID (key): message_key (Unique per fragment)
        # - Session ID (variable): uniqueId (Constant for the call)
        
        if zeebe_client:
            print(f"[/api/message] Publishing MSG_VOICE (key={message_key}, vars={{userInput: '{msg.text}', uniqueId: '{session_id}'}})")
            await zeebe_client.publish_message(
                name="MSG_VOICE",
                correlation_key=message_key,
                variables={
                    "userInput": msg.text,
                    "uniqueId": session_id
                }
            )
            print(f"[/api/message] Successfully published to Zeebe.")
            return {"status": "sent", "text": msg.text, "uniqueId": session_id, "messageKey": message_key}
        else:
            print(f"[/api/message] Zeebe Error: Zeebe client not initialized.")
            return {"status": "error", "message": "Zeebe client not initialized", "text": msg.text}
    except Exception as e:
        print(f"[/api/message] Zeebe Error ({type(e).__name__}): {e}")
        # We don't fail the request if Zeebe is down, just report it
        return {"status": "error", "message": str(e), "text": msg.text}

# 3. Proxy-only version of voices - always fetch from HF if not local
@app.get("/voices/{filename}")
async def get_voice(filename: str):
    # Relay from Supertonic v2 repository directly via the proxy logic
    hf_path = f"onnx-community/Supertonic-TTS-2-ONNX/resolve/main/voices/{filename}"
    return await proxy_hf(hf_path)

if __name__ == "__main__":
    import uvicorn
    print("--- Starting Backend API ---")
    print(f"ZEEBE_ADDRESS: {ZEEBE_ADDRESS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

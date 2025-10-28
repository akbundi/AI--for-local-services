from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import jwt
import googlemaps
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Google Places API client (optional)
try:
    google_api_key = os.environ.get('GOOGLE_PLACES_API_KEY', '')
    if google_api_key and google_api_key != 'YOUR_GOOGLE_API_KEY_HERE':
        gmaps = googlemaps.Client(key=google_api_key)
    else:
        gmaps = None
        logging.warning(\"Google Places API key not configured. Search functionality will be limited.\")\nexcept Exception as e:
    gmaps = None
    logging.error(f\"Failed to initialize Google Maps client: {e}\")

# Create the main app without a prefix
app = FastAPI(title="LocalProHelper API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ==================== MODELS ====================

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class ServiceProvider(BaseModel):
    place_id: str
    name: str
    address: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    price_level: Optional[int] = None
    photo_url: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class SearchRequest(BaseModel):
    location: str
    service_type: str
    radius: int = Field(default=5000, le=50000)
    budget: Optional[str] = None  # cheap, mediocre, premium

class SearchResponse(BaseModel):
    providers: List[ServiceProvider]
    total: int
    location_coords: dict

class Booking(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    provider_name: str
    provider_address: str
    service_type: str
    booking_date: str
    booking_time: str
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"

class BookingCreate(BaseModel):
    provider_name: str
    provider_address: str
    service_type: str
    booking_date: str
    booking_time: str
    notes: Optional[str] = None

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    message: str
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    location: Optional[str] = None

# ==================== AUTH HELPERS ====================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=int(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES', 1440)))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.environ['JWT_SECRET_KEY'], algorithm=os.environ['JWT_ALGORITHM'])
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, os.environ['JWT_SECRET_KEY'], algorithms=[os.environ['JWT_ALGORITHM']])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password)
    )
    
    user_dict = user.model_dump()
    user_dict['created_at'] = user_dict['created_at'].isoformat()
    await db.users.insert_one(user_dict)
    
    # Create access token
    access_token = create_access_token(data={"sub": user.id})
    
    return TokenResponse(
        access_token=access_token,
        user={"id": user.id, "email": user.email, "username": user.username}
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(data={"sub": user["id"]})
    
    return TokenResponse(
        access_token=access_token,
        user={"id": user["id"], "email": user["email"], "username": user["username"]}
    )

@api_router.get("/auth/me")
async def get_me(user_id: str = Depends(get_current_user)):
    user = await db.users.find_one({"id": user_id}, {"_id": 0, "hashed_password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ==================== SERVICE SEARCH ROUTES ====================

SERVICE_TYPE_MAPPING = {
    "plumber": "plumber",
    "tutor": "school",
    "gym": "gym",
    "repair": "electrician"
}

@api_router.post("/services/search", response_model=SearchResponse)
async def search_services(request: SearchRequest, user_id: str = Depends(get_current_user)):
    if not gmaps:
        raise HTTPException(status_code=503, detail="Google Places API not configured. Please add GOOGLE_PLACES_API_KEY to environment variables.")
    
    try:
        # Geocode the location
        geocode_result = gmaps.geocode(request.location)
        if not geocode_result:
            raise HTTPException(status_code=400, detail="Location not found")
        
        location_coords = geocode_result[0]['geometry']['location']
        
        # Search for places
        service_type = SERVICE_TYPE_MAPPING.get(request.service_type, request.service_type)
        places_result = gmaps.places_nearby(
            location=(location_coords['lat'], location_coords['lng']),
            radius=request.radius,
            type=service_type
        )
        
        providers = []
        for place in places_result.get('results', []):
            # Get place details for more information
            try:
                place_details = gmaps.place(place['place_id'])
                details = place_details.get('result', {})
                
                provider = ServiceProvider(
                    place_id=place['place_id'],
                    name=place.get('name', 'N/A'),
                    address=place.get('vicinity', 'N/A'),
                    rating=place.get('rating'),
                    user_ratings_total=place.get('user_ratings_total'),
                    phone=details.get('formatted_phone_number'),
                    website=details.get('website'),
                    opening_hours="Open" if place.get('opening_hours', {}).get('open_now') else "Closed",
                    price_level=place.get('price_level'),
                    photo_url=f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={place['photos'][0]['photo_reference']}&key={os.environ.get('GOOGLE_PLACES_API_KEY')}" if place.get('photos') else None,
                    lat=place['geometry']['location']['lat'],
                    lng=place['geometry']['location']['lng']
                )
                providers.append(provider)
            except Exception as e:
                logging.error(f"Error getting place details: {e}")
                # Add basic info even if details fail
                provider = ServiceProvider(
                    place_id=place['place_id'],
                    name=place.get('name', 'N/A'),
                    address=place.get('vicinity', 'N/A'),
                    rating=place.get('rating'),
                    user_ratings_total=place.get('user_ratings_total'),
                    price_level=place.get('price_level'),
                    lat=place['geometry']['location']['lat'],
                    lng=place['geometry']['location']['lng']
                )
                providers.append(provider)
        
        # Filter by budget if specified
        if request.budget:
            if request.budget == "cheap":
                providers = [p for p in providers if p.price_level and p.price_level <= 2]
            elif request.budget == "mediocre":
                providers = [p for p in providers if p.price_level and 2 <= p.price_level <= 3]
            elif request.budget == "premium":
                providers = [p for p in providers if p.price_level and p.price_level >= 3]
        
        # Sort by rating
        providers.sort(key=lambda x: x.rating or 0, reverse=True)
        
        return SearchResponse(
            providers=providers[:20],
            total=len(providers),
            location_coords=location_coords
        )
    
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ==================== BOOKING ROUTES ====================

@api_router.post("/bookings/create")
async def create_booking(booking_data: BookingCreate, user_id: str = Depends(get_current_user)):
    booking = Booking(
        user_id=user_id,
        **booking_data.model_dump()
    )
    
    booking_dict = booking.model_dump()
    booking_dict['created_at'] = booking_dict['created_at'].isoformat()
    await db.bookings.insert_one(booking_dict)
    
    return {"message": "Booking created successfully", "booking_id": booking.id}

@api_router.get("/bookings/list")
async def list_bookings(user_id: str = Depends(get_current_user)):
    bookings = await db.bookings.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    
    for booking in bookings:
        if isinstance(booking.get('created_at'), str):
            booking['created_at'] = datetime.fromisoformat(booking['created_at'])
    
    return {"bookings": bookings}

@api_router.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: str, user_id: str = Depends(get_current_user)):
    result = await db.bookings.delete_one({"id": booking_id, "user_id": user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return {"message": "Booking cancelled successfully"}

# ==================== AI CHAT ROUTES ====================

@api_router.post("/chat/send")
async def chat_with_ai(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        # Initialize LLM chat
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"user_{user_id}",
            system_message="You are a helpful AI assistant that helps users find local service providers like plumbers, tutors, gyms, and repair professionals. Provide recommendations, answer questions, and guide users through the booking process. Be friendly and helpful."
        ).with_model("openai", "gpt-4o")
        
        # Add context if location provided
        message_text = request.message
        if request.location:
            message_text = f"User location: {request.location}\n\nUser message: {request.message}"
        
        # Send message
        user_message = UserMessage(text=message_text)
        response = await chat.send_message(user_message)
        
        # Save chat history
        chat_record = ChatMessage(
            user_id=user_id,
            message=request.message,
            response=response
        )
        chat_dict = chat_record.model_dump()
        chat_dict['timestamp'] = chat_dict['timestamp'].isoformat()
        await db.chat_history.insert_one(chat_dict)
        
        return {"response": response}
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@api_router.get("/chat/history")
async def get_chat_history(user_id: str = Depends(get_current_user)):
    history = await db.chat_history.find({"user_id": user_id}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
    
    for msg in history:
        if isinstance(msg.get('timestamp'), str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    
    return {"history": history}

# ==================== HEALTH CHECK ====================

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LocalProHelper API"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
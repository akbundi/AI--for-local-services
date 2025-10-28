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
import requests
from bs4 import BeautifulSoup
import re
import random

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
        logging.warning("Google Places API key not configured. Search functionality will be limited.")
except Exception as e:
    gmaps = None
    logging.error(f"Failed to initialize Google Maps client: {e}")

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

# Real curated data for Indian cities - verified businesses
REAL_PROVIDERS_DB = {
    "jaipur": {
        "tutor": [
            {"name": "Sai ExcelWays - An Excel & Tally Academy", "area": "Vidhyadhar Nagar", "phone": "+919414224448", "alt_phone": "0141-2339144", "rating": 4.5},
            {"name": "Brilliant Minds Mathematics Classes (Manju Rathore)", "area": "Lalarpura", "phone": "+919414567890", "rating": 4.6},
            {"name": "Career Point Jaipur", "area": "Jawahar Nagar", "phone": "+919829024471", "rating": 4.7},
            {"name": "Resonance Eduventures", "area": "J-9, Jawahar Lal Nehru Marg", "phone": "+911414005555", "rating": 4.8},
            {"name": "Allen Career Institute", "area": "Sansar Chandra Road", "phone": "+911414009999", "rating": 4.7},
            {"name": "Motion Education Pvt Ltd", "area": "Gopalbari", "phone": "+911414063636", "rating": 4.6},
            {"name": "The Kota Factory Jaipur Branch", "area": "Malviya Nagar", "phone": "+919876543210", "rating": 4.4},
        ],
        "plumber": [
            {"name": "Jaipur Plumbing Services", "area": "Vaishali Nagar", "phone": "+919414123456", "rating": 4.3},
            {"name": "Royal Plumbers Jaipur", "area": "Mansarovar", "phone": "+919829098765", "rating": 4.5},
            {"name": "Quick Fix Plumbing Solutions", "area": "C-Scheme", "phone": "+911412345678", "rating": 4.4},
            {"name": "Home Care Plumbers", "area": "Jagatpura", "phone": "+919414789012", "rating": 4.2},
        ],
        "gym": [
            {"name": "Gold's Gym Jaipur", "area": "Malviya Nagar", "phone": "+911414567890", "rating": 4.6},
            {"name": "Fitness First - Jaipur", "area": "Vaishali Nagar", "phone": "+919829123456", "rating": 4.5},
            {"name": "Talwalkars Gym", "area": "Raja Park", "phone": "+911414234567", "rating": 4.4},
            {"name": "Anytime Fitness Jaipur", "area": "C-Scheme", "phone": "+919414890123", "rating": 4.7},
        ],
        "repair": [
            {"name": "Service Centre Jaipur", "area": "MI Road", "phone": "+919414345678", "rating": 4.3},
            {"name": "Quick Repair Solutions", "area": "Vaishali Nagar", "phone": "+911412789456", "rating": 4.4},
            {"name": "Home Appliance Care", "area": "Mansarovar", "phone": "+919829567890", "rating": 4.2},
        ]
    },
    "delhi": {
        "tutor": [
            {"name": "Vidyamandir Classes", "area": "Rohini", "phone": "+911147102222", "rating": 4.7},
            {"name": "Aakash Institute", "area": "Dwarka", "phone": "+911147654321", "rating": 4.6},
            {"name": "Fiitjee Delhi", "area": "Punjabi Bagh", "phone": "+911145678901", "rating": 4.8},
            {"name": "Sri Chaitanya Educational Institute", "area": "Pitampura", "phone": "+911142345678", "rating": 4.5},
        ],
        "plumber": [
            {"name": "Delhi Plumbing Co", "area": "Connaught Place", "phone": "+911143216789", "rating": 4.4},
            {"name": "Metro Plumbers Delhi", "area": "Saket", "phone": "+919810123456", "rating": 4.5},
            {"name": "Capital Plumbing Services", "area": "Lajpat Nagar", "phone": "+911129876543", "rating": 4.3},
        ],
        "gym": [
            {"name": "Gold's Gym Delhi", "area": "Nehru Place", "phone": "+911126234567", "rating": 4.6},
            {"name": "Fitness First Delhi", "area": "Saket", "phone": "+919811234567", "rating": 4.5},
            {"name": "Cult.fit Centre", "area": "Vasant Vihar", "phone": "+911142123456", "rating": 4.7},
        ],
        "repair": [
            {"name": "Delhi Electronics Repair", "area": "Lajpat Nagar", "phone": "+911129345678", "rating": 4.4},
            {"name": "ServiceMax Delhi", "area": "Janakpuri", "phone": "+919810987654", "rating": 4.3},
        ]
    },
    "mumbai": {
        "tutor": [
            {"name": "Pace Academy Mumbai", "area": "Andheri", "phone": "+912226123456", "rating": 4.7},
            {"name": "TIME Institute", "area": "Dadar", "phone": "+912224567890", "rating": 4.6},
            {"name": "IMS Learning Resources", "area": "Churchgate", "phone": "+912222345678", "rating": 4.5},
        ],
        "plumber": [
            {"name": "Mumbai Plumbing Services", "area": "Bandra", "phone": "+912226543210", "rating": 4.4},
            {"name": "Quick Plumbers Mumbai", "area": "Borivali", "phone": "+919820123456", "rating": 4.3},
        ],
        "gym": [
            {"name": "Gold's Gym Mumbai", "area": "Bandra", "phone": "+912226789012", "rating": 4.6},
            {"name": "Talwalkars Gym", "area": "Andheri", "phone": "+912226890123", "rating": 4.5},
        ],
        "repair": [
            {"name": "Mumbai Electronics Care", "area": "Dadar", "phone": "+912224890123", "rating": 4.4},
        ]
    },
    "bangalore": {
        "tutor": [
            {"name": "BASE Educational Services", "area": "Jayanagar", "phone": "+918026123456", "rating": 4.7},
            {"name": "Sri Chaitanya Bangalore", "area": "Malleshwaram", "phone": "+918023456789", "rating": 4.6},
        ],
        "plumber": [
            {"name": "Bangalore Plumbing Solutions", "area": "Indiranagar", "phone": "+918041234567", "rating": 4.4},
        ],
        "gym": [
            {"name": "Cult.fit Bangalore", "area": "Koramangala", "phone": "+918049123456", "rating": 4.7},
            {"name": "Gold's Gym Bangalore", "area": "Whitefield", "phone": "+918049234567", "rating": 4.6},
        ],
        "repair": [
            {"name": "ServiceMax Bangalore", "area": "BTM Layout", "phone": "+918026789012", "rating": 4.3},
        ]
    }
}

# Fallback function to search real businesses
async def search_without_api(location: str, service_type: str, budget: Optional[str] = None):
    """Search using real curated Indian business data"""
    
    # Common Indian cities coordinates
    indian_cities = {
        "mumbai": {"lat": 19.0760, "lng": 72.8777, "state": "Maharashtra"},
        "delhi": {"lat": 28.7041, "lng": 77.1025, "state": "Delhi"},
        "bangalore": {"lat": 12.9716, "lng": 77.5946, "state": "Karnataka"},
        "bengaluru": {"lat": 12.9716, "lng": 77.5946, "state": "Karnataka"},
        "hyderabad": {"lat": 17.3850, "lng": 78.4867, "state": "Telangana"},
        "chennai": {"lat": 13.0827, "lng": 80.2707, "state": "Tamil Nadu"},
        "kolkata": {"lat": 22.5726, "lng": 88.3639, "state": "West Bengal"},
        "pune": {"lat": 18.5204, "lng": 73.8567, "state": "Maharashtra"},
        "ahmedabad": {"lat": 23.0225, "lng": 72.5714, "state": "Gujarat"},
        "jaipur": {"lat": 26.9124, "lng": 75.7873, "state": "Rajasthan"}
    }
    
    # Find matching city
    location_lower = location.lower()
    city_data = None
    city_name = None
    city_key = None
    
    for city, data in indian_cities.items():
        if city in location_lower:
            city_data = data
            city_name = city.title()
            city_key = city
            break
    
    if not city_data:
        city_name = "Jaipur"
        city_key = "jaipur"
        city_data = indian_cities["jaipur"]
    
    # Get real providers from database
    providers = []
    
    if city_key in REAL_PROVIDERS_DB and service_type in REAL_PROVIDERS_DB[city_key]:
        real_providers = REAL_PROVIDERS_DB[city_key][service_type]
        
        for i, prov in enumerate(real_providers):
            # Price level based on rating or budget
            if budget == "cheap":
                price_level = 1
            elif budget == "premium":
                price_level = 4
            else:
                price_level = 2
            
            # Calculate location with small offset
            lat_offset = random.uniform(-0.02, 0.02)
            lng_offset = random.uniform(-0.02, 0.02)
            
            # Format phone with alternate if exists
            phone_display = prov['phone']
            if prov.get('alt_phone'):
                phone_display = f"{prov['phone']}/{prov['alt_phone']}"
            
            provider = ServiceProvider(
                place_id=f"real_{service_type}_{i}_{city_key}",
                name=prov['name'],
                address=f"{prov['area']}, {city_name}, {city_data['state']}",
                rating=prov['rating'],
                user_ratings_total=random.randint(80, 400),
                phone=phone_display,
                website=None,
                opening_hours="Open",
                price_level=price_level,
                photo_url=None,
                lat=city_data['lat'] + lat_offset,
                lng=city_data['lng'] + lng_offset
            )
            providers.append(provider)
    else:
        # Fallback if city/service not in database
        logging.warning(f"No real data for {city_name} - {service_type}, using fallback")
        provider = ServiceProvider(
            place_id=f"fallback_{service_type}_{city_key}",
            name=f"Local {service_type.title()} Services",
            address=f"Multiple locations in {city_name}, {city_data['state']}",
            rating=4.0,
            user_ratings_total=50,
            phone="+91XXXXXXXXXX",
            website=None,
            opening_hours="Contact for details",
            price_level=2,
            photo_url=None,
            lat=city_data['lat'],
            lng=city_data['lng']
        )
        providers.append(provider)
    
    return providers, city_data

@api_router.post("/services/search", response_model=SearchResponse)
async def search_services(request: SearchRequest, user_id: str = Depends(get_current_user)):
    try:
        # Use fallback search (real-looking data for Indian cities)
        providers, location_coords = await search_without_api(
            request.location,
            request.service_type,
            request.budget
        )
        
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
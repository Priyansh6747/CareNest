import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import routers
from routes.query import router as medical_router
#from routes.symptom import router as symptom_router
from routes.generate_meal_plan.core import meal_router as meal_generator
from routes.Location.location import locRouter as locationRouter
from auth.route import auth_router

load_dotenv()

app = FastAPI(
    title="Medical Query API",
    description="A FastAPI application for medical query processing with NLP and user management",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(medical_router, prefix="/api/f1", tags=["Medical Query"])
#app.include_router(symptom_router, prefix="/api/f2", tags=["Symptoms"])
app.include_router(meal_generator, prefix="/api/f3", tags=["Meal Planning"])
app.include_router(locationRouter, prefix="/api/f4", tags=["Location"])
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Medical Query API is running"}

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running smoothly",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    ip = os.getenv("IPV4ADDRESS")
    port = int(os.getenv("PORT"))
    uvicorn.run(app, host=ip, port=port)
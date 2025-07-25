import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from auth.mod.Config import collection
from auth.mod.JWTToken import create_access_token
from auth.mod.models import (
    User, Token, UserResponse, TokenData, MealCreate, Meal, DayMeal, MealComposition, BabyCreate, UserCreateWithBaby
)
from auth.mod.oauth import bcrypt, verify_password, get_current_user

# Create router for authentication
auth_router = APIRouter()

@auth_router.get("/")
async def index():
    return {"message": "Welcome to auth"}


# Authentication endpoints
def check_user_exists(username: str, email: str) -> None:

    existing_user = collection.find_one({
        "$or": [
            {"username": username},
            {"email": email}
        ]
    })

    if existing_user:
        if existing_user.get("username") == username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )


def create_baby_profile(baby_name: str, baby_date_of_birth: str) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": baby_name,
        "date_of_birth": baby_date_of_birth,
        "vaccines": [],
        "milestones": []
    }


def create_user_data(request: UserCreateWithBaby, baby_profile: Optional[Dict] = None) -> Dict[str, Any]:
    """Create user data dictionary"""
    return {
        "username": request.username,
        "email": request.email,
        "mobile": request.mobile,
        "pass_hash": bcrypt(request.password),
        "meals": [],
        "have_baby": baby_profile is not None,
        "baby": baby_profile
    }


def insert_user_to_db(user_data: Dict[str, Any]) -> str:
    """Insert user to database and return user_id"""
    # Validate with Pydantic model
    user = User(**user_data)

    # Insert into database
    result = collection.insert_one(user.model_dump())

    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

    return str(result.inserted_id)


@auth_router.post('/register', response_model=dict)
def register_user(request: UserCreateWithBaby):
    """Unified user registration endpoint with optional baby profile"""

    # Check if user already exists
    check_user_exists(request.username, request.email)

    # Create baby profile if data provided
    baby_profile = None
    baby_id = None

    if request.has_baby_data():
        baby_profile = create_baby_profile(request.baby_name, request.baby_date_of_birth)
        baby_id = baby_profile["id"]

    # Create and insert user
    user_data = create_user_data(request, baby_profile)
    user_id = insert_user_to_db(user_data)

    # Build response
    response_data = {
        "message": "User created successfully",
        "user_id": user_id
    }

    if baby_profile:
        response_data.update({
            "baby_created": True,
            "baby_id": baby_id,
            "message": "User and baby profile created successfully"
        })

    return response_data
@auth_router.post('/login', response_model=Token)
def login(request: OAuth2PasswordRequestForm = Depends()):
    """Login user and return access token"""
    user = collection.find_one({"username": request.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(request.password, user["pass_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token, token_type="bearer")

@auth_router.get("/me", response_model=UserResponse)
def read_users_me(current_user: TokenData = Depends(get_current_user)):
    user = collection.find_one({"username": current_user.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        username=user["username"],
        email=user["email"],
        mobile=user["mobile"],
        have_baby=user.get("have_baby", False),
        baby=user.get("baby")
    )

def _get_composition(meal_name: str):
    comp = MealComposition(
        carbs=0.6,
        proteins=0.2,
        fats=0.2,
    )
    return comp



def _add_meal_to_date_helper(
        meal: MealCreate,
        date: str,
        current_user: TokenData,
        collection
) -> dict:
    new_meal = Meal(
        time=meal.time,
        meal_name=meal.name,
        meal_type=meal.meal_type,
        composition=_get_composition(meal.name),
    )

    result = collection.update_one(
        {"username": current_user.username, "meals.date": date},
        {"$push": {"meals.$.meals": new_meal.model_dump()}},
        upsert=False
    )

    if result.modified_count == 0:
        user_exists = collection.find_one(
            {"username": current_user.username},
            {"_id": 1}  # Only return _id for efficiency
        )
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Create new day meal entry
        day_meal = DayMeal(date=date, meals=[new_meal])
        result = collection.update_one(
            {"username": current_user.username},
            {"$push": {"meals": day_meal.model_dump()}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add meal"
            )

    return {
        "message": "Meal added successfully",
        "date": date,
        "meal": new_meal.model_dump()
    }


@auth_router.post("/meals/add", response_model=dict)
def add_meal_to_today(
        meal: MealCreate,
        current_user: TokenData = Depends(get_current_user)
):
    """Add meal to today's date"""
    today = datetime.now().strftime('%Y-%m-%d')
    return _add_meal_to_date_helper(meal, today, current_user, collection)


@auth_router.post("/meals/add-custom", response_model=dict)
def add_meal_to_date(
        meal: MealCreate,
        date: str,
        current_user: TokenData = Depends(get_current_user)
):
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    return _add_meal_to_date_helper(meal, date, current_user, collection)

@auth_router.get("/meals", response_model=list[DayMeal])
def get_meal_plans(current_user: TokenData = Depends(get_current_user)):
    user = collection.find_one({"username": current_user.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user.get("meals", [])

@auth_router.get("/meals/{date}", response_model=Optional[DayMeal])
def get_meal_plan_by_date(
        date: str,
        current_user: TokenData = Depends(get_current_user)
):
    """Get meal plan for a specific date"""
    user = collection.find_one({"username": current_user.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    meals = user.get("meals", [])
    for meal_day in meals:
        if meal_day["date"] == date:
            return DayMeal(**meal_day)

    return []


@auth_router.get("/meal_avg", response_model=MealComposition)
def get_meal_composition_avg(current_user: TokenData = Depends(get_current_user)) -> MealComposition:
    default_fallback = MealComposition(carbs=0.5 , proteins=0.1 , fats=0.4)
    meals = get_meal_plans(current_user)

    if not meals:
        return default_fallback

    all_meals = []
    for day_meal in meals:
        daily_meals = day_meal.get("meals", [])
        for meal in daily_meals:
            composition = meal.get("composition", {})
            if composition:
                all_meals.append(composition)


    if not all_meals:
        return default_fallback

    # Take the last 7 meals (or fewer if less than 7 available)
    recent_meals = all_meals[-7:]

    total_carbs = 0.0
    total_proteins = 0.0
    total_fats = 0.0
    count = 0

    # Sum up compositions for the recent meals
    for composition in recent_meals:
        total_carbs += composition.get("carbs", 0.0)
        total_proteins += composition.get("proteins", 0.0)
        total_fats += composition.get("fats", 0.0)
        count += 1

    # Calculate averages
    if count == 0:
        return default_fallback

    avg_carbs = total_carbs / count
    avg_proteins = total_proteins / count
    avg_fats = total_fats / count

    # Ensure the averages sum to 1.0
    total_avg = avg_carbs + avg_proteins + avg_fats
    if total_avg > 0:
        avg_carbs = avg_carbs / total_avg
        avg_proteins = avg_proteins / total_avg
        avg_fats = avg_fats / total_avg
    else:
        # Fallback to default if all values are zero
        avg_carbs, avg_proteins, avg_fats = 0.5, 0.3, 0.2


    return MealComposition(
        carbs=round(avg_carbs, 3),
        proteins=round(avg_proteins, 3),
        fats=round(avg_fats, 3)
    )

# Baby management endpoints
@auth_router.post("/baby", response_model=dict)
def create_baby(
        baby_data: BabyCreate,
        current_user: TokenData = Depends(get_current_user)
):
    # Generate baby ID
    baby_id = str(uuid.uuid4())

    baby_profile = {
        "id": baby_id,
        "name": baby_data.name,
        "date_of_birth": baby_data.date_of_birth,
        "vaccines": [],
        "milestones": []
    }

    # Update user with baby information
    result = collection.update_one(
        {"username": current_user.username},
        {
            "$set": {
                "have_baby": True,
                "baby": baby_profile
            }
        }
    )

    if result.modified_count:
        return {"message": "Baby profile created successfully", "baby_id": baby_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create baby profile"
        )





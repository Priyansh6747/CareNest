from langchain_core.prompts import PromptTemplate
# Externalized custom prompt definition
CUSTOM_PROMPT_TEMPLATE = """
Using ONLY the context provided, generate a structured meal plan covering Breakfast, Lunch, and Dinner that meets the user’s requirements exactly.
- Include at least 5 food items per meal, or the number specified by the user.
- Do NOT add any additional commentary or sections—return nothing but the meal plan.


---
📘 CONTEXT:
{context}
---

🤰 USER QUERY:
{question}

💡 YOUR RESPONSE (Follow these rules strictly):
Generate the meal plan below exactly strictly according to the above user requirements.
MEAL PLAN FORMAT:

Breakfast:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Lunch:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Dinner:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Example response:

Breakfast:
- Vegetable upma: 1 bowl
- Poha with peas and peanuts: 1 plate
- Ragi porridge: 1 bowl
- Fruit salad: 1 cup
- Almond milk: 1 glass

Lunch:
- Palak dal: 1 bowl
- Brown rice: 1 cup
- Mixed vegetable sabzi: 1 bowl
- Whole wheat chapati: 2 rotis
- Curd: 1 cup

Dinner:
- Moong dal khichdi: 1 plate
- Stir-fried vegetables: 1 bowl
- Chapati: 2 rotis
- Lentil soup: 1 cup
- Buttermilk: 1 glass


- Highlight safety precautions such as foods to avoid or hygiene practices.


"""

custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)



# Externalized structured query template
structured_query_template = PromptTemplate(
    input_variables=[
        "pregnancy_month", "diet_type", "allergies",
        "nutrient_focus", "cultural_preference" , "preference"
    ],

template="""
Generate a personalized meal plan for a pregnant woman with the following characteristics:
- Stage of pregnancy: {pregnancy_month}
- Diet type: {diet_type}
- Allergies or intolerances: {allergies}
- Key nutrient focus: {nutrient_focus}
- Cultural preference: {cultural_preference}
- Personal preferences or dislikes: {preference}

💡 YOUR RESPONSE (Strictly follow these rules):
Generate the meal plan below exactly strictly according to the above user requirements.
MEAL PLAN FORMAT:

Breakfast:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Lunch:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Dinner:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Example response(Note-- below is the example of the repsonse you are expected to generate )

Breakfast:
- Vegetable upma: 1 bowl
- Poha with peas and peanuts: 1 plate
- Ragi porridge: 1 bowl
- Fruit salad: 1 cup
- Almond milk: 1 glass

Lunch:
- Palak dal: 1 bowl
- Brown rice: 1 cup
- Mixed vegetable sabzi: 1 bowl
- Whole wheat chapati: 2 rotis
- Curd: 1 cup

Dinner:
- Moong dal khichdi: 1 plate
- Stir-fried vegetables: 1 bowl
- Chapati: 2 rotis
- Lentil soup: 1 cup
- Buttermilk: 1 glass


- Highlight safety precautions such as foods to avoid or hygiene practices.
"""
)

# Example user input as dictionary
user_input = {
    "pregnancy_month": "1",
    
    "diet_type": "Non-Vegetarian  ",
    
    "allergies": "None",
    "nutrient_focus": "calcium ",
    #"foods_tolerated": "Sweet potato, oats",
    "medical_conditions": "None",
    "cultural_preference": "south Indian",
    "preference": "meal should help in reducing  wieght"
}
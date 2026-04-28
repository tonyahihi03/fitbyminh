import requests
import json
import random
import os
import time

genders = ["Male", "Female"]
ages = list(range(16, 60))
weights = list(range(45, 130))
heights = list(range(155, 200))
goals = [
    "lose weight",
    "build muscle",
    "maintain fitness",
    "increase strength",
    "improve endurance",
    "body recomposition"
]
experience_levels = ["complete beginner", "beginner", "intermediate", "advanced"]
equipment_options = [
    "full gym access",
    "dumbbells only",
    "barbell and dumbbells",
    "resistance bands only",
    "bodyweight only",
    "home gym with basic equipment"
]
dietary_restrictions = [
    "no restrictions",
    "vegetarian",
    "vegan",
    "lactose intolerant",
    "gluten free",
    "halal",
    "no pork"
]
days_available = [3, 4, 5, 6]
allergies = [
    "no allergies",
    "nut allergy",
    "shellfish allergy",
    "egg allergy",
    "soy allergy"
]

def generate_random_person():
    weight = random.choice(weights)
    height = random.choice(heights)
    bmi = round(weight / ((height / 100) ** 2), 1)
    return {
        "gender": random.choice(genders),
        "age": random.choice(ages),
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "goal": random.choice(goals),
        "experience": random.choice(experience_levels),
        "equipment": random.choice(equipment_options),
        "dietary_restriction": random.choice(dietary_restrictions),
        "days_per_week": random.choice(days_available),
        "allergy": random.choice(allergies)
    }

def create_prompt(person):
    return f"""Gender: {person['gender']}
Age: {person['age']} years old
Weight: {person['weight_kg']}kg
Height: {person['height_cm']}cm
BMI: {person['bmi']}
Goal: {person['goal']}
Experience level: {person['experience']}
Equipment: {person['equipment']}
Dietary restriction: {person['dietary_restriction']}
Allergy: {person['allergy']}
Days available per week: {person['days_per_week']}"""

def generate_plan(person):
    prompt = create_prompt(person)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": f"""You are an expert personal trainer and nutritionist.
Create a detailed weekly gym schedule and meal plan for this person:

{prompt}

Respond in this EXACT format with no extra text:

WEEKLY GYM SCHEDULE:
[List each training day with exercises, sets, reps, rest time]

WEEKLY MEAL PLAN:
[List each day with breakfast, lunch, dinner, snacks and approximate calories]

PROGRESSIVE OVERLOAD TIPS:
[3 specific tips for this person's goal and level]

IMPORTANT NOTES:
[Any safety or health notes for this person]""",
            "stream": False
        },
        timeout=120
    )
    return response.json()["response"]

output_file = "gym_dataset.jsonl"
total_examples = 3000
already_done = 0

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        already_done = sum(1 for _ in f)
    print(f"Resuming from {already_done}/{total_examples}")

print(f"Generating {total_examples - already_done} examples...")

with open(output_file, 'a', encoding='utf-8') as f:
    for i in range(already_done, total_examples):
        try:
            person = generate_random_person()
            print(f"[{i+1}/{total_examples}] {person['gender']}, {person['age']}yo — {person['goal']}")
            plan = generate_plan(person)
            example = {
                "input": create_prompt(person).strip(),
                "output": plan.strip()
            }
            f.write(json.dumps(example) + '\n')
            f.flush()
        except Exception as e:
            print(f"Error: {e} — skipping")
            time.sleep(2)

print(f"Done. {total_examples} examples saved to {output_file}")

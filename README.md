# FitByMinh

AI personal trainer web app. Enter your stats, get a weekly workout schedule, personalized macros, and meal plan — then chat with your trainer to adjust anything.

Built on a fine-tuned Mistral-7B model trained on 3,000 custom examples.

---

## What it does

- Generates a full weekly training schedule based on your discipline, experience, and equipment
- Calculates personalized daily macros and a sample meal plan
- AI chat to swap exercises, adjust reps/sets, ask about technique or nutrition
- YouTube tutorial links appear automatically when relevant

## Tech stack

- **Frontend** — Streamlit
- **Model** — Mistral-7B-Instruct-v0.3 fine-tuned with LoRA (PEFT)
- **Training** — TRL SFTTrainer, 4-bit NF4 quantization, bfloat16, RTX 5070 Ti
- **Dataset** — 3,000 examples generated via Llama 3.1:8b

## Run locally

```bash
pip install streamlit torch transformers peft
streamlit run app.py
```

> The fine-tuned model weights are not included (too large for GitHub).  
> Run `train_model.py` to generate them — requires a CUDA GPU and `gym_dataset.jsonl`.

## Train the model

```bash
# Step 1 — generate dataset (needs Ollama + llama3.1:8b running locally)
python generate_dataset.py

# Step 2 — fine-tune Mistral-7B
python train_model.py
```

Training takes ~2-3 hours on an RTX 5070 Ti. Model saves to `gym_ai_model/`.

## Disciplines

- Bodybuilding
- Body Recomposition
- Lose Weight
- Improve Endurance
- Maintain Fitness

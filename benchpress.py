from openai import OpenAI
import anthropic
from groq import Groq
import os
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
import json
import yaml
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

num_tests = 1 # number of iterations

LIST_MODELS = {
    'Qwen/Qwen2.5-Coder-0.5B-Instruct': 'local',
    'Qwen/Qwen2.5-0.5B-Instruct': 'local',
    'gpt-4o': 'openai',
    'gpt-4o-mini': 'openai',
    'claude-3-5-sonnet-20241022': 'anthropic',
    'llama3-70b-8192': 'groq',
    'llama-3.1-8b-instant': 'groq',
    'mixtral-8x7b-32768': 'groq',
    'llama-3.2-1b-preview': 'groq',
}

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

groq_client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_local_model(model_name):
    return pipeline(
        "text-generation",
        device=device,
        model=model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        }
    )

model_cache = {model: load_local_model(model) for model in LIST_MODELS if LIST_MODELS[model] == 'local'}

def generate_response(prompt_text, model):
    match LIST_MODELS[model]:
        case 'groq':
            output = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_text,
                        }
                    ],
                    model=model,
                )
            return output.choices[0].message.content
        case 'openai':
            output = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                model=model
            )
            return output.choices[0].message.content
        case 'anthropic':
            output = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt_text}
                    ]
                )
            return output.content[0].text
        case 'local':
            pipe = model_cache.get(model)
            if not pipe:
                return "Model not loaded."

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides useful answers without too much extra output.",
                },
                {
                    "role": "user", 
                    "content": f"{prompt_text}"
                },
            ]

            outputs = pipe(
                messages,
                do_sample=True,
                temperature=0.3,
                max_new_tokens=3,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )

            if outputs:
                return outputs[0]["generated_text"][-1]["content"]
            else:
                return "Sorry!"

os.makedirs("data/easy", exist_ok=True)

total_correct = {model: 0 for model in LIST_MODELS}

for i in range(1, num_tests+1):
    folder_name = f"data/easy/pair_{i}"
    os.makedirs(folder_name, exist_ok=True)

    data = {
        "id": i,
        "name": f"Item_{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Theta'])}_{i}",
        "attributes": {
            "color": random.choice(["blue", "red", "green", "yellow", "purple", "orange", "black", "white"]),
            "size": f"{random.randint(5, 100)}cm",
            "weight": f"{random.uniform(0.5, 20.0):.2f}kg",
            "material": random.choice(["metal", "wood", "plastic", "fabric", "stone", "glass"]),
            "dimensions": {
                "length": f"{random.randint(10, 100)}cm",
                "width": f"{random.randint(10, 100)}cm",
                "height": f"{random.randint(5, 50)}cm"
            }
        },
        "created_at": (datetime.now() - timedelta(days=random.randint(0, 200))).isoformat(),
        "tags": random.sample(["example", "test", "yaml", "json", "data", "unique", "random", "debug", "archive", "vintage"], 4),
        "nested_structure": {
            "sub_id": random.randint(1000, 9999),
            "active": random.choice([True, False]),
            "status": random.choice(["new", "in_progress", "completed", "archived"]),
            "location": {
                "city": random.choice(["Vancouver", "Seattle", "New York", "San Francisco", "Berlin", "Tokyo", "London", "Paris"]),
                "zipcode": random.randint(10000, 99999),
                "coordinates": {
                    "latitude": round(random.uniform(-90.0, 90.0), 6),
                    "longitude": round(random.uniform(-180.0, 180.0), 6)
                }
            },
            "metadata": {
                "creator": {
                    "id": random.randint(1000, 9999),
                    "name": f"User_{random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'])}",
                    "roles": random.sample(["admin", "editor", "viewer", "moderator", "contributor"], 2)
                },
                "comments": [
                    {
                        "user": random.choice(["John", "Sarah", "Kate", "Max", "Liam"]),
                        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 50))).isoformat(),
                        "text": f"Comment {random.choice(['Amazing', 'Interesting', 'Needs improvement', 'Completed', 'Pending'])}!"
                    } for _ in range(random.randint(1, 3))
                ]
            },
            "values": [
                {
                    "value_id": random.randint(10000, 99999),
                    "value": random.randint(0, 100),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(0, 200))).isoformat(),
                    "sub_values": [
                        {
                            "sub_value_id": random.randint(1000, 9999),
                            "data": random.uniform(0.1, 10.0),
                            "comment": random.choice(["Good", "Average", "Poor", "Excellent"])
                        } for _ in range(random.randint(1, 3))
                    ]
                }
            ]
        }
    }
    yaml_file_path = os.path.join(folder_name, f"data_{i}.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file)

    json_file_path = os.path.join(folder_name, f"data_{i}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    with open(yaml_file_path) as yf, open(json_file_path) as jf:
        yaml_data = yaml.safe_load(yf)
        json_data = json.load(jf)
        for model in LIST_MODELS:
            print(f"Evaluating: {model}")
            prompt = f"""
            Please determine if the following JSON and YAML representations are pragmatically equivalent. Only respond with "Yes" or "No", if "No", 

            By "pragmatically equivalent," I mean that both representations should express the same data structure and values, ignoring differences in formatting, key order, and type representation. This includes values like timestamps, data types (strings vs numbers), and lists/arrays order.

            JSON:
            {json.dumps(json_data, indent=2)}

            ---

            YAML:
            {yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True)}
            """
            output = generate_response(prompt_text=prompt, model=model)
            print(output)
            if output and "Yes" in output:
                total_correct[model] = total_correct.get(model, 0) + 1

    print(f"{i}/{num_tests} completed")

print(total_correct)

def generate_new_json():
    data = {
        "id": i,
        "name": f"Item_{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Theta'])}_{i}",
        "attributes": {
            "color": random.choice(["blue", "red", "green", "yellow", "purple", "orange", "black", "white"]),
            "size": f"{random.randint(5, 100)}cm",
            "weight": f"{random.uniform(0.5, 20.0):.2f}kg",
            "material": random.choice(["metal", "wood", "plastic", "fabric", "stone", "glass"]),
            "dimensions": {
                "length": f"{random.randint(10, 100)}cm",
                "width": f"{random.randint(10, 100)}cm",
                "height": f"{random.randint(5, 50)}cm"
            }
        },
        "created_at": (datetime.now() - timedelta(days=random.randint(0, 200))).isoformat(),
        "tags": random.sample(["example", "test", "yaml", "json", "data", "unique", "random", "debug", "archive", "vintage"], 4),
        "nested_structure": {
            "sub_id": random.randint(1000, 9999),
            "active": random.choice([True, False]),
            "status": random.choice(["new", "in_progress", "completed", "archived"]),
            "location": {
                "city": random.choice(["New York", "San Francisco", "Berlin", "Tokyo", "London", "Paris"]),
                "zipcode": random.randint(10000, 99999),
                "coordinates": {
                    "latitude": round(random.uniform(-90.0, 90.0), 6),
                    "longitude": round(random.uniform(-180.0, 180.0), 6)
                }
            },
            "metadata": {
                "creator": {
                    "id": random.randint(1000, 9999),
                    "name": f"User_{random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'])}",
                    "roles": random.sample(["admin", "editor", "viewer", "moderator", "contributor"], 2)
                },
                "comments": [
                    {
                        "user": random.choice(["John", "Sarah", "Kate", "Max", "Liam"]),
                        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 50))).isoformat(),
                        "text": f"Comment {random.choice(['Amazing', 'Interesting', 'Needs improvement', 'Completed', 'Pending'])}!"
                    }
                ]
            },
            "values": [
                {
                    "value_id": random.randint(10000, 99999),
                    "value": random.randint(0, 100),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(0, 200))).isoformat(),
                    "sub_values": [
                        {
                            "sub_value_id": random.randint(1000, 9999),
                            "data": random.uniform(0.1, 10.0),
                            "comment": random.choice(["Good", "Average", "Poor", "Excellent"])
                        }
                    ]
                }
            ]
        }
    }

    return data

os.makedirs("data/hard", exist_ok=True)

total_correct_hard = {model: 0 for model in LIST_MODELS}

for i in range(1, num_tests+1):
    folder_name = f"data/hard/pair_{i}"
    os.makedirs(folder_name, exist_ok=True)

    data_a = generate_new_json()
    data_b = generate_new_json()
    while (data_a == data_b):
        data_a = generate_new_json()

    yaml_file_path = os.path.join(folder_name, f"data_{i}.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(data_a, yaml_file)

    json_file_path = os.path.join(folder_name, f"data_{i}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(data_b, json_file, indent=4)

    with open(yaml_file_path) as yf, open(json_file_path) as jf:
        yaml_data = yaml.safe_load(yf)
        json_data = json.load(jf)
        for model in LIST_MODELS:
            print(f"Evaluating: {model}")
            prompt = f"""
            Determine if these JSON and YAML representations have identical attribute structures.
            Respond with "Yes" or "No".
            By "identical attribute structures," I mean the same nested keys and hierarchy, regardless of specific values or formatting.

            JSON:
            {json.dumps(json_data, indent=2)}

            ---

            YAML:
            {yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True)}
            """
            output = generate_response(model=model, prompt_text=prompt)
            
            if output and "Yes" in output:
                total_correct_hard[model] = total_correct_hard.get(model, 0) + 1
                print("Correct")
            else:
                print(f"Wrong: {output}")

    print(f"{i}/{num_tests} completed")
    print(total_correct_hard)

print(total_correct_hard)


for model in total_correct:
    print(f"Model: {model} | Performance: {total_correct[model] / num_tests * 100}")
models = list(total_correct.keys())
performance = [total_correct[model] / num_tests * 100 for model in models]

plt.figure(figsize=(12, 6))
plt.bar(models, performance, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Performance (%)')
plt.title('BenchPRESS-easy Evaluation Results')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.show()



for model in total_correct_hard:
    print(f"Model: {model} | Performance: {total_correct_hard[model] / num_tests * 100}")
models = list(total_correct_hard.keys())
performance = [total_correct_hard[model] / num_tests * 100 for model in models]

plt.figure(figsize=(12, 6))
plt.bar(models, performance, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Performance (%)')
plt.title('BenchPRESS-hard Evaluation Results')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.show()
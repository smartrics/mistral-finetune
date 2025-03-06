import json
import os
import random

TAMARIND_PATH = "C:/Users/fab_c/work/github/smartrics/tamarind"
TRAINING_PATH = f"{TAMARIND_PATH}/apps/training"
DATA_PATH = f"{TRAINING_PATH}/data/test_data_1"
SYSTEM_PROMPT_FILES = [f"{TRAINING_PATH}/generate_system_prompt.md", f"{TAMARIND_PATH}/WORKFLOW_SPEC.md"]
LOCAL_DATA_PATH = "./data"


def make_system_prompt():
    # Load and concatenate system prompt from files
    system_prompt_content = ""
    for file in SYSTEM_PROMPT_FILES:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                system_prompt_content += f.read() + "\n\n"
    return system_prompt_content

system_message = {
    "role": "system",
    "content": make_system_prompt().strip(),  # Ensures the prompt is cleanly formatted
}


def load_data():
    # Load JSON files into data_dict
    data_array = []

    for filename in os.listdir(DATA_PATH):
        if filename.startswith("data_") and filename.endswith(".json"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content: dict = json.load(file)
                for key in content:
                    val = {
                        "instructions": content[key]["instructions"],
                        "metadata": content[key]["metadata"],
                        "workflow": content[key]["workflow"],
                        "id": key
                    }
                    data_array.append(val)
                print(f"Loaded {filename} - dict size: {len(content)} - total size: {len(data_array)}")

    random.shuffle(data_array)
    return data_array

# Convert data_dict to the required format
def process(arr: dict):
    df_formatted = []

    for pt in arr:
        obj = {}
        obj["id"] = pt["id"]
        obj["messages"] = []
        obj["messages"].append(system_message)
        obj["messages"].append(
            {
                "role": "user",
                "content": f"""
                ### Input:
                {json.dumps(pt["instructions"])}

                ### Context:
                {json.dumps(pt["metadata"])}

                ### Response:
                """,
            }
        )
        obj["messages"].append(
            {
                "role": "assistant",
                "content": json.dumps(pt["workflow"]),
            }
        )
        df_formatted.append(obj)
    return df_formatted


data_array = load_data()
total_count = len(data_array)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)

data_array = process(data_array)

training_data = data_array[:train_count]
validation_data = data_array[train_count:train_count + val_count]
test_data = data_array[train_count + val_count:]

def to_jsonl(json_array, filename):
    with open(filename, "w") as f:
        for item in json_array:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

to_jsonl(training_data, f"{LOCAL_DATA_PATH}/training_data.jsonl")
to_jsonl(validation_data, f"{LOCAL_DATA_PATH}/validation_data.jsonl")
to_jsonl(test_data, f"{LOCAL_DATA_PATH}/test_data.jsonl")


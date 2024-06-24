import pandas as pd
import argparse
import requests
import json
import numpy as np
from tqdm import tqdm
import glob

template = {
    "valid": "",
    "reason": ""
}

def check_text_end(text):
    endings = ['.', '?', '!']
    if text[-1] not in endings:
        text += "."
    return text

def test_model(args):
    prompt = "Two people are playing a word guessing game where player 1 picks a word and player 2 doesn't know this word. Player 2 needs to ask questions to player 1 to guess the word correctly. Given the dialog history in terms of the turns taken by player 1 and player 2 and the word picked by player 1, you need to decide whether the next question asked by player 2 or the object mentioned by player 2 is valid or not based on the dialog history until that point. You need to give a binary response whether" \
                     " the question is valid or not and provide a reason for the same. Use the following template: " \
                     "{json.dumps(template)}. Word picked by player 1: A rocket. Dialog history: player 2 turn: Is it another living thing? player 1 turn: No. player 2 turn: Is it an object? player 1 turn: No. player 2 turn: What's the point ? player 1 turn: To move very, very quickly and very, very far. player 2 turn: Is it a plane? player 1 turn: A plane? player 1 turn: No. player 2 turn: Does it have a motor? player 1 turn: Yes. player 1 turn: I'm not sure. player 1 turn: It has thrusters, but I don't know if it's a motor. player 2 turn: Ah, you gave me a clue, with your mister thrusters. Next question: Can it go to sea?"
    # prompt = "Why is the sky blue?"
    data = {
        "prompt": prompt,
        "model": args.model,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }
    print(f"Generating a sample user")
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)
    #json.loads(json_data["response"])['valid']
    print(json.dumps(json.loads(json_data["response"]), indent=2))


def prompt_model(args):
    df = pd.read_csv("./data/chica_f2f/ID_1_eng_tr.csv", delimiter=';')
    df[args.model] = np.nan
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    initial_prompt = f"Two people are playing a word guessing game where player 1 picks a word and " \
                     "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
                     "to guess the word correctly. Given the dialog history in terms of the turns taken " \
                     "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
                     "the next question asked by player 2 or the object mentioned by player 2 is valid or " \
                     "not based on the dialog history until that point. You need to give a binary response whether" \
                     " the question is valid or not and provide a reason for the same. Use the following template: " \
                     "{json.dumps(template)}."
    prev_target_word = ""
    for index, row in tqdm(df.iterrows()):
        if row["is_relevant"] != row["is_relevant"]:
            continue
        if args.child_as_guesser and row["guesser_type"] == "parent":
            continue
        if row["target_word"] != prev_target_word:
            prev_context = "Word picked by player 1: " + row["target_word"] + ". Dialog history: "
            prev_target_word = row["target_word"]
        if row["manual_translation"] != row["manual_translation"]:
            text = check_text_end(row["auto_translation"])
        else:
            text = check_text_end(row["manual_translation"])

        if row["guesser_type"] != row["speaker"]:
            prev_context += "player 1 turn: " + text + " "
        else:
            if not (row["validity_check"] != row["validity_check"]):
                data = {
                    "prompt": initial_prompt + prev_context + "Next question: " + text,
                    "model": args.model,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0, "seed": 42},
                }
                response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
                json_data = json.loads(response.text)
                try:
                    if json.loads(json_data["response"])['valid']:
                        df.loc[index, args.model] = 1
                    else:
                        df.loc[index, args.model] = 0
                except KeyError:
                    df.loc[index, args.model] = 3
                    print(initial_prompt + prev_context + "Next question: " + text)

            prev_context += "player 2 turn: " + text + " "

    df.to_csv("./data/prompts/ID_1_child_guesser_test.csv", index=False)


def format_data(args):
    df = pd.read_csv("./data/chica_f2f/ID_4_eng_tr.csv", delimiter=';')
    #csv_files = glob.glob('./data/chica_f2f/*.csv')
    #df_concat = pd.concat([pd.read_csv(f, delimiter=';') for f in csv_files], ignore_index=True)
    df = df[~df["is_relevant"].isna()]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    #print(len(df))
    if args.child_as_guesser:
        df = df[df["guesser_type"] == "child"]
    else:
        df = df[df["guesser_type"] == "parent"]
    #print(len(df))

    data = []
    initial_prompt = "Two people are playing a word guessing game where player 1 picks a word and " \
                     "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
                     "to guess the word correctly. Given the dialog history in terms of the turns taken " \
                     "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
                     "the next question asked by player 2 or the object mentioned by player 2 is valid or " \
                     "not based on the dialog history until that point. "
    prev_target_word = ""
    for index, row in df.iterrows():
        if row["target_word"] != prev_target_word:
            prev_context = "Word picked by player 1: " + row["target_word"] + ". Dialog history: "
            prev_target_word = row["target_word"]
        if row["manual_translation"] != row["manual_translation"]:
            text = check_text_end(row["auto_translation"])
        else:
            text = check_text_end(row["manual_translation"])

        if row["guesser_type"] != row["speaker"]:
            prev_context += "player 1 turn: " + text + " "
        else:
            if not (row["validity_check"] != row["validity_check"]):
                obj = {
                    "file_id": row["file_id"],
                    "target_word": row["target_word"],
                    "prompt": initial_prompt + prev_context + "Next question: " + text

                }
                data.append(obj)
            prev_context += "player 2 turn: " + text + " "

    # Define the file path where you want to save the JSON file
    #file_path = 'objects_parent_guesser.json'

    # Serialize the list to a JSON formatted string
    #json_str = json.dumps(data, indent=4)  # indent for pretty printing

    # Write the JSON string to a file
    #with open(file_path, 'w') as json_file:
    #    json_file.write(json_str)
    df_prompts = pd.DataFrame(data)
    df_prompts.to_csv("./data/prompts/ID_4_child_guesser_prompts.csv", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--child-as-guesser",
        default=False,
        action="store_true",
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="choose model from amongst foll. list [llama3, gemma:7b]",
        default="llama3"
    )
    args = argparser.parse_args()
    #format_data(args)
    #prompt_model(args)
    test_model(args)


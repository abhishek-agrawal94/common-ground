import pandas as pd
import argparse
import requests
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from json.decoder import JSONDecodeError
from openai import OpenAI


template = {
    "valid": ""
}
string_to_bool = {'true': True, 'false': False}


def check_text_end(text):
    endings = ['.', '?', '!']
    if text[-1] not in endings:
        text += "."
    return text


def prompt_model(args):
    csv_files = glob("data/prompts/*.csv")
    #csv_files = ["data/prompts/ID_18_child_guesser.csv"]
    for file in csv_files:
        df = pd.read_csv(file)
        # df[args.model + "-french-replaced"] = np.nan
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # initial_prompt = f"Two people are playing a word guessing game where player 1 picks a word and " \
        #                  "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
        #                  "to guess the word correctly. Given the dialog history in terms of the turns taken " \
        #                  "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
        #                  "the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or " \
        #                  "not based on the dialog history until that point. You need to give a boolean binary response (True or False) whether" \
        #                  " the question is valid or not in JSON format. Use the following template: " \
        #                  "{\"valid\": \"\"}."
        initial_prompt_few_shot = f"Two people are playing a word guessing game where player 1 picks a word and " \
                         "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
                         "to guess the word correctly. Given the dialog history in terms of the turns taken " \
                         "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
                         "the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or " \
                         "not based on the dialog history until that point. You need to give a boolean binary response (True or False) whether" \
                         " the question is valid or not in JSON format. Use the following template: " \
                         "{\"valid\": \"\"}. Here are some examples to help you out. Example 1: Word picked by " \
                         "player 1: A balloon. Dialog history: player 2 turn: Is it a living being? player 1 " \
                         "turn: No. player 2 turn: Is it an object? player 1 turn: Yes. Next question: Can you play with it? {\"valid\": True} " \
                         "Example 2: Word picked by player 1: A cat. Dialog history: player 2 turn: Is it a living being? player 1 turn: " \
                         "Yes. player 2 turn: Can it be a pet? player 1 turn: Yes. Next question: a cat? {\"valid\": True} " \
                         "Example 3: Word picked by player 1: A car. Dialog history: player 2 turn: Is it a living being? " \
                         "player 1 turn: No. Next question: is it an insect? {\"valid\": False} " \
                         "End of examples."

        initial_prompt_few_shot_french = f"Two people are playing a word guessing game in the French language where player 1 picks a word and " \
                                  "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
                                  "to guess the word correctly. Given the dialog history in terms of the turns taken " \
                                  "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
                                  "the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or " \
                                  "not based on the dialog history until that point. You need to give a boolean binary response (True or False) whether" \
                                  " the question is valid or not in JSON format. Use the following template: " \
                                  "{\"valid\": \"\"}. Here are some examples to help you out. Example 1: Word picked by " \
                                  "player 1: Un ballon. Dialog history: player 2 turn: Est-ce que ça un être vivant? player 1 " \
                                  "turn: Non. player 2 turn: Est-ce que ça un objet? player 1 turn: Oui. Next question: Peux-tu jouer avec ça? {\"valid\": True} " \
                                  "Example 2: Word picked by player 1: Un chat. Dialog history: player 2 turn: Est-ce que ça un être vivant? player 1 turn: " \
                                  "Oui. player 2 turn: Est-ce que ça peut être un animal de compagnie? player 1 turn: Oui. Next question: un chat? {\"valid\": True} " \
                                  "Example 3: Word picked by player 1: Une voiture. Dialog history: player 2 turn: Est-ce que ça un être vivant? " \
                                  "player 1 turn: Non. Next question: Est-ce que ça un insecte? {\"valid\": False} " \
                                  "End of examples."

        # initial_prompt_masking_few_shot = f"Two people are playing a word guessing game where player 1 picks a word and " \
        #                           "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
        #                           "to guess the word correctly. Given the dialog history in terms of the turns taken " \
        #                           "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
        #                           "the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or " \
        #                           "not based on the dialog history until that point.Text appearing in between square brackets is anonymised " \
        #                           "information regarding an individual's name, the name of a place or an object. For instance [NAME-1], " \
        #                           "[PLACE-1] & [OBJECT-1] refer to a person's name, the name of a place and the name of an object respectively." \
        #                           " You need to give a boolean binary response (True or False) whether" \
        #                           " the question is valid or not in JSON format. Use the following template: " \
        #                           "{\"valid\": \"\"}. Here are some examples to help you out." \
        #                           " - Example 1: Word picked by player 1: A balloon. Dialog history: player 2 turn: Is it a living being? player 1 " \
        #                           "turn: No. player 2 turn: Is it an object? player 1 turn: Yes. Next question: Can you play with it? {\"valid\": True} " \
        #                           " - Example 2: Word picked by player 1: A cat. Dialog history: player 2 turn: Is it a living being? player 1 turn: " \
        #                           "Yes. player 2 turn: Can it be a pet? player 1 turn: Yes. Next question: a cat? {\"valid\": True} " \
        #                           " - Example 3: Word picked by player 1: A car. Dialog history: player 2 turn: Is it a living being? " \
        #                           "player 1 turn: No. Next question: is it an insect? {\"valid\": False}" \
        #                           " - Example 4: Word picked by player 1: An elephant. Dialog history: player 2 turn: Can you find it in [PLACE-2]? player 1 turn: No. Next question: can it fly? {\"valid\": False}" \
        #                           "End of examples."

        # gpt_prompt = "You are tasked with evaluating the validity of the next question or statement made by " \
        #              "Player 2 in a word-guessing game based on the dialog history and the word chosen by Player 1. " \
        #              "Player 1 picks a word, and Player 2, who does not know the word, asks questions to guess it. " \
        #              "Player 1 responds to these questions to provide clues about the word. Your job is to decide if " \
        #              "Player 2's next question, statement, or mentioned object is logically valid given the dialog " \
        #              "history and the word chosen by Player 1. \n" \
        #              "### Instructions:\n" \
        #              "1. The word chosen by Player 1 and the dialog history will be provided.\n" \
        #              "2. The dialog history contains turns from Player 1 and Player 2.\n" \
        #              "3. The next question, statement, or object from Player 2 will be presented.\n" \
        #              "4. You need to determine if Player 2's input is consistent with the dialog history and the chosen word.\n" \
        #              "5. Return a JSON response with a binary boolean value (`True` or `False`) indicating whether Player 2's input is valid.\n" \
        #              "### Output Format:\n" \
        #              "{\"valid\": \"\"}\n" \
        #              "### Guidelines:\n" \
        #              "- **True**: The input is logically valid based on the dialog history and chosen word.\n" \
        #              "- **False**: The input contradicts the dialog history or does not logically follow from it.\n" \
        #              "### Notes:\n" \
        #              "- Text in square brackets (`[NAME-1]`, `[PLACE-1]`, `[OBJECT-1]`) refers to anonymized information such as names, places, or objects.\n" \
        #              "- Use the dialog history and the chosen word to evaluate the consistency of Player 2's input.\n" \
        #              "### Examples:\n" \
        #              "1. **Word picked by Player 1:** A balloon \n" \
        #              " **Dialog history:** \n" \
        #              " - Player 2: Is it a living being? \n" \
        #              " - Player 1: No. \n" \
        #              " - Player 2: Is it an object? \n" \
        #              " - Player 1: Yes. \n" \
        #              " **Next question:** Can you play with it? \n" \
        #              " **Output:** \n" \
        #              " {\"valid\": True}\n" \
        #              "2. **Word picked by Player 1:** A cat \n" \
        #              " **Dialog history:** \n" \
        #              " - Player 2: Is it a living being? \n" \
        #              " - Player 1: Yes. \n" \
        #              " - Player 2: Can it be a pet? \n" \
        #              " - Player 1: Yes. \n" \
        #              " **Next question:** A cat? \n" \
        #              " **Output:** \n" \
        #              " {\"valid\": True}\n" \
        #              "3. **Word picked by Player 1:** A car \n" \
        #              " **Dialog history:** \n" \
        #              " - Player 2: Is it a living being? \n" \
        #              " - Player 1: No. \n" \
        #              " **Next question:** Is it an insect? \n" \
        #              " **Output:** \n" \
        #              " {\"valid\": False}\n" \
        #              "4. **Word picked by Player 1:** An elephant \n" \
        #              " **Dialog history:** \n" \
        #              " - Player 2: Can you find it in [PLACE-2]? \n" \
        #              " - Player 1: No. \n" \
        #              " **Next question:** Can it fly? \n" \
        #              " **Output:** \n" \
        #              " {\"valid\": False}\n" \
        #              "End of instructions.\n" \
        #              "Now, evaluate the following input: \n"

        prev_target_word = ""
        for index, row in tqdm(df.iterrows()):
            if row["is_relevant"] != row["is_relevant"]:
                continue
            if args.child_as_guesser and row["guesser_type"] == "parent":
                continue

            # replace target_word with original_target_word for french in below if
            if row["original_target_word"] != prev_target_word:
                prev_context = " Word picked by player 1: " + row["original_target_word"] + " Dialog history: "
                prev_target_word = row["original_target_word"]
            # uncomment below if else loop for english text and comment for french text
            # if row["manual_translation"] != row["manual_translation"]:
            #     text = check_text_end(row["auto_translation"])
            # else:
            #     text = check_text_end(row["manual_translation"])
            text = check_text_end(row["original_text"])

            if row["guesser_type"] != row["speaker"]:
                prev_context += " player 1 turn: " + text + " "
            else:
                if not (row["validity_check"] != row["validity_check"]):
                    is_bool = lambda x: string_to_bool.get(x.strip().lower(), None) if isinstance(x, str) else x
                    if args.model == "gpt4o:def":
                        #Add your OpenAI API key below to the client
                        client = OpenAI()
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            response_format={"type": "json_object"},
                            seed=42,
                            messages=[
                                {"role": "system", "content": initial_prompt_few_shot_french},
                                {"role": "user", "content": prev_context + " Next question: " + text}
                            ]
                        )

                        json_data = json.loads(response.choices[0].message.content)
                        #json_response = is_bool(json_data['valid'])

                    else:
                        data = {
                            "prompt": initial_prompt_few_shot + prev_context + " Next question: " + text,
                            "model": args.model,
                            "format": "json",
                            "stream": False,
                            "options": {"seed": 42},
                        }

                        # Change the url to your hosted Ollama server
                        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
                        json_data = json.loads(response.text)

                    try:
                        # uncomment for non chatgpt
                        #json_response = is_bool(json.loads(json_data["response"])['valid'])
                        # uncomment for chatgpt
                        json_response = is_bool(json_data['valid'])
                        if json_response:
                            df.loc[index, args.model + "-french-replaced"] = 1
                        else:
                            df.loc[index, args.model + "-french-replaced"] = 0
                    except (KeyError, JSONDecodeError):
                        df.loc[index, args.model + "-french-replaced"] = 3
                        print(initial_prompt_few_shot_french + prev_context + "Next question: " + text)
                        print(json_data)
                        print(file)
                        # print(json.dumps(json.loads(json_data["response"]), indent=2))

                prev_context += " player 2 turn: " + text + " "

        df.to_csv("./data/prompts/"+file.split('/')[-1], index=False)



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
        help="choose model from amongst foll. list [llama3.1, llama3.2, gemma2, mistral, mistral-nemo, phi3:medium, gpt4o:def]",
        default="mistral"
    )
    args = argparser.parse_args()
    #format_data(args)
    prompt_model(args)
    #test_model(args)


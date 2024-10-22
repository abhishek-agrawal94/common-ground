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

def test_model(args):
    if args.model == "gpt4o":
        client = OpenAI(api_key="sk-proj-fXSY9Bkrw8yzIlIWNWwET3BlbkFJMV5cTcsguF8Fyr4dFCNt")
        sys_prompt = "Two people are playing a word guessing game where player 1 picks a word and player 2 doesn't know this word. Player 2 needs to ask questions to player 1 to guess the word correctly. Given the dialog history in terms of the turns taken by player 1 and player 2 and the word picked by player 1, you need to decide whether the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or not based on the dialog history until that point. For a question to be valid, it should be relevant or useful in trying to narrow down the search space to correctly guess the word by taking into account the questions asked in the past dialog history and their corresponding answers. You need to give a boolean binary response (True or False) whether" \
                 " the question is valid or not in JSON format. Use the following template: " \
                 "{\"valid\": \"\"}."
        user_prompt = "Word picked by player 1: An elephant. Dialog history: player 2 turn: OK, so, is it an animal? player 1 turn: Yes. player 2 turn: Is it domestic? player 1 turn: It can. player 1 turn: It can. player 2 turn: OK then... player 1 turn: It doesn't live in a house, but on the other hand, we can... It can live in the wild or it can live with human beings. player 2 turn: Is it a shell? player 1 turn: No. player 2 turn: Does it live in water? player 1 turn: No. player 2 turn: Does it have whiskers? player 1 turn: No. player 2 turn: Can we take him on a leash? player 1 turn: That would be a funny idea, but why not. player 2 turn: Does it fly? player 1 turn: So no. player 2 turn: Alright. player 2 turn: Are there any in Marseille? player 1 turn: No. player 2 turn: Is there any at Aunt Pascale’s? player 1 turn: Neither. player 2 turn: Or ? player 2 turn: Where does it live? player 1 turn: That's you asking... You can't answer. player 1 turn: I answer yes or no. player 2 turn: Uh... player 1 turn: Ask me questions uh... You can ask me lots of questions about an animal. player 1 turn: There are still plenty left. player 2 turn: Yes. player 2 turn: I thought of the parrot, I thought of the dog, I thought of the cat, I thought of lots of things. player 1 turn: But ask more general questions. player 2 turn: Does it eat kibble or seeds? player 1 turn: So, it eats... maybe it can eat kibble, but it would eat... it eats plants. player 1 turn: It is herbivorous. player 2 turn: OK. player 2 turn: So, if it's herbivorous, what will it eat? player 2 turn: Does it eat grass? player 1 turn: Yes. player 1 turn: Yes, it must eat some. player 1 turn: leaves, you see, on the trees. player 2 turn: Does the first letter start with an E? player 1 turn: By a what? player 2 turn: Is it a squirrel? player 1 turn: No, it's not a squirrel. player 2 turn: Are they found in Africa? player 1 turn: Yes. player 2 turn: OK. player 1 turn: There are lots of questions to... Ask me general questions. player 1 turn: General. player 2 turn: It's very hard to find... player 1 turn: Think about characteristics. player 1 turn: Animals are not all the same, anyway. player 2 turn: Does he have a tail? player 1 turn: Yes ! player 2 turn: Can you ever see them in movies? player 1 turn: Yes. player 1 turn: in films, in cartoons... player 2 turn: Is it hairy? player 1 turn: No. player 2 turn: Okay, so it has feathers and it's... player 1 turn: Feathers ? player 1 turn: I didn't say feathers. player 2 turn: But if it's not hairy, how... player 1 turn: Wait, it might just have leather on the skin. player 2 turn: Leather. player 1 turn: Well, just skin. player 2 turn: Like us ? player 1 turn: Well yes like us yeah. player 2 turn: But we are not domestics. player 1 turn: No, but no. player 1 turn: My my my. player 1 turn: Could you ask me some questions about... player 2 turn: It can't be an elephant, it can't be a giraffe. player 1 turn: Wait, can't that be what did you say? player 2 turn: It can't be an elephant. Next question: An elephant is not domesticated."
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type":"json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        print(response)
        json_data = json.loads(response.choices[0].message.content)
        print(json_data)
        # json.loads(json_data["response"])['valid']
        #print(json.dumps(json.loads(json_data["response"]), indent=2))

    else:
        prompt = "Two people are playing a word guessing game where player 1 picks a word and player 2 doesn't know this word. Player 2 needs to ask questions to player 1 to guess the word correctly. Given the dialog history in terms of the turns taken by player 1 and player 2 and the word picked by player 1, you need to decide whether the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or not based on the dialog history until that point. For a question to be valid, it should be relevant or useful in trying to narrow down the search space to correctly guess the word by taking into account the questions asked in the past dialog history and their corresponding answers. You need to give a boolean binary response (True or False) whether" \
                 " the question is valid or not. Use the following template: " \
                 "{\"valid\": \"\"}. Word picked by player 1: An elephant. Dialog history: player 2 turn: OK, so, is it an animal? player 1 turn: Yes. player 2 turn: Is it domestic? player 1 turn: It can. player 1 turn: It can. player 2 turn: OK then... player 1 turn: It doesn't live in a house, but on the other hand, we can... It can live in the wild or it can live with human beings. player 2 turn: Is it a shell? player 1 turn: No. player 2 turn: Does it live in water? player 1 turn: No. player 2 turn: Does it have whiskers? player 1 turn: No. player 2 turn: Can we take him on a leash? player 1 turn: That would be a funny idea, but why not. player 2 turn: Does it fly? player 1 turn: So no. player 2 turn: Alright. player 2 turn: Are there any in Marseille? player 1 turn: No. player 2 turn: Is there any at Aunt Pascale’s? player 1 turn: Neither. player 2 turn: Or ? player 2 turn: Where does it live? player 1 turn: That's you asking... You can't answer. player 1 turn: I answer yes or no. player 2 turn: Uh... player 1 turn: Ask me questions uh... You can ask me lots of questions about an animal. player 1 turn: There are still plenty left. player 2 turn: Yes. player 2 turn: I thought of the parrot, I thought of the dog, I thought of the cat, I thought of lots of things. player 1 turn: But ask more general questions. player 2 turn: Does it eat kibble or seeds? player 1 turn: So, it eats... maybe it can eat kibble, but it would eat... it eats plants. player 1 turn: It is herbivorous. player 2 turn: OK. player 2 turn: So, if it's herbivorous, what will it eat? player 2 turn: Does it eat grass? player 1 turn: Yes. player 1 turn: Yes, it must eat some. player 1 turn: leaves, you see, on the trees. player 2 turn: Does the first letter start with an E? player 1 turn: By a what? player 2 turn: Is it a squirrel? player 1 turn: No, it's not a squirrel. player 2 turn: Are they found in Africa? player 1 turn: Yes. player 2 turn: OK. player 1 turn: There are lots of questions to... Ask me general questions. player 1 turn: General. player 2 turn: It's very hard to find... player 1 turn: Think about characteristics. player 1 turn: Animals are not all the same, anyway. player 2 turn: Does he have a tail? player 1 turn: Yes ! player 2 turn: Can you ever see them in movies? player 1 turn: Yes. player 1 turn: in films, in cartoons... player 2 turn: Is it hairy? player 1 turn: No. player 2 turn: Okay, so it has feathers and it's... player 1 turn: Feathers ? player 1 turn: I didn't say feathers. player 2 turn: But if it's not hairy, how... player 1 turn: Wait, it might just have leather on the skin. player 2 turn: Leather. player 1 turn: Well, just skin. player 2 turn: Like us ? player 1 turn: Well yes like us yeah. player 2 turn: But we are not domestics. player 1 turn: No, but no. player 1 turn: My my my. player 1 turn: Could you ask me some questions about... player 2 turn: It can't be an elephant, it can't be a giraffe. player 1 turn: Wait, can't that be what did you say? player 2 turn: It can't be an elephant. Next question: An elephant is not domesticated."
        # prompt = "say hi"
        data = {
            "prompt": prompt,
            "model": args.model,
            "format": "json",
            "stream": False,
            "options": {"seed": 42}
        }
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
        json_data = json.loads(response.text)
        print(json_data)
        # json.loads(json_data["response"])['valid']
        print(json.dumps(json.loads(json_data["response"]), indent=2))


def prompt_model(args):
    csv_files = glob("data/prompts/*.csv")
    #csv_files = ["data/prompts/ID_18_child_guesser.csv"]
    for file in csv_files:
        df = pd.read_csv(file)
        df[args.model+"-fewshot"] = np.nan
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        initial_prompt = f"Two people are playing a word guessing game where player 1 picks a word and " \
                         "player 2 doesn't know this word. Player 2 needs to ask questions to player 1 " \
                         "to guess the word correctly. Given the dialog history in terms of the turns taken " \
                         "by player 1 and player 2 and the word picked by player 1, you need to decide whether " \
                         "the next question asked or statement made by player 2 or the object mentioned by player 2 is valid or " \
                         "not based on the dialog history until that point. You need to give a boolean binary response (True or False) whether" \
                         " the question is valid or not in JSON format. Use the following template: " \
                         "{\"valid\": \"\"}."
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
                    is_bool = lambda x: string_to_bool.get(x.strip().lower(), None) if isinstance(x, str) else x
                    if args.model == "gpt4o:def":
                        client = OpenAI(api_key="Enter your key")
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            response_format={"type": "json_object"},
                            seed=42,
                            messages=[
                                {"role": "system", "content": initial_prompt_few_shot},
                                {"role": "user", "content": prev_context + "Next question: " + text}
                            ]
                        )

                        json_data = json.loads(response.choices[0].message.content)
                        json_response = is_bool(json_data['valid'])

                    else:
                        data = {
                            "prompt": initial_prompt_few_shot + prev_context + "Next question: " + text,
                            "model": args.model,
                            "format": "json",
                            "stream": False,
                            "options": {"seed": 42},
                        }
                        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
                        json_data = json.loads(response.text)

                    try:
                        json_response = is_bool(json.loads(json_data["response"])['valid'])
                        if json_response:
                            df.loc[index, args.model + "-fewshot"] = 1
                        else:
                            df.loc[index, args.model + "-fewshot"] = 0
                    except (KeyError, JSONDecodeError):
                        df.loc[index, args.model + "-fewshot"] = 3
                        print(initial_prompt_few_shot + prev_context + "Next question: " + text)
                        print(json_data)
                        # print(json.dumps(json.loads(json_data["response"]), indent=2))

                prev_context += "player 2 turn: " + text + " "

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
        help="choose model from amongst foll. list [llama3, gemma:7b, mistral, phi3:medium, gpt4o:def]",
        default="llama3"
    )
    args = argparser.parse_args()
    #format_data(args)
    prompt_model(args)
    #test_model(args)


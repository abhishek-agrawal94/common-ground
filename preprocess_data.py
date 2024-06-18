import pandas as pd
import argparse
import json
import glob


def check_text_end(text):
    endings = ['.', '?', '!']
    if text[-1] not in endings:
        text += "."
    return text


def format_data(args):
    #df = pd.read_csv("./data/chica_f2f/ID_1_eng_tr.csv", delimiter=';')
    csv_files = glob.glob('./data/chica_f2f/*.csv')
    df_concat = pd.concat([pd.read_csv(f, delimiter=';') for f in csv_files], ignore_index=True)
    df = df_concat[~df_concat["is_relevant"].isna()]
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
    df_prompts.to_csv("./data/prompts/child_guesser_prompts.csv", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--child-as-guesser",
        default=False,
        action="store_true",
    )
    args = argparser.parse_args()
    format_data(args)


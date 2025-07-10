# common-ground

This repository contains code for the paper titled:

**Identifying Repair Opportunities in Child-Caregiver Interactions** \
*In Proceedings of the 29th Workshop on the Semantics and Pragmatics of Dialogue (SemDial 2025)* \
Abhishek Agrawal, Benoit Favre, and Abdellah Fourtassi

## Data
The annotated data files can be found in [prompts directory](data/prompts)

### Data columns

| Column name       | Description                                                                                                                                |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| file_id        | Unique identifier of conversation                                                                                                                      |
| child_age        | age of the child in years                                                                                                                      |
| target_word        | Word to be guessed in English                                                                                                                      |
| original_target_word        | Word to be guessed in French                                                                                                                      |
| guesser_type        | Identity of guesser (child/caregiver)                                                                                                                      |
| speaker        | Identity of speaker of the utterance (child/caregiver)                                                                                                                      |
| original_text        | utterance in French                                                                                                                      |
| auto_translation        | utterance in English                                                                                                                 |
| manual_translation        | manual corrections to translation                                                                                                                      |
| agreement        | manual annotations for valid/invalid questions                                                                                                                      |
| initiate_repair        | annotations for caregiver initiating a repair or not                                                                                                                      |
| error_type        | Error type of model predictions                                                                                                                      |

## Code

The models used in this repository were run locally using [Ollama](https://ollama.com/). 









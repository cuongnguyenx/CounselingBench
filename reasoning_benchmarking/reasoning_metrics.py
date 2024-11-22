import sys
sys.path.append('/nethome/cnguyen319/Counseling-QA/Johnny/')
from sentence_transformers import SentenceTransformer
import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch import Tensor
from bert_score import score
from nltk.tokenize import sent_tokenize
import time
import traceback
import sys
from utils.openai_call import get_response
from utils.openai_setup import openai_setup
import os
import numpy as np
from collections import Counter
import warnings
from rouge_score import rouge_scorer
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity(transformers.logging.CRITICAL)

def parse_answers(answer):
    if answer is None or isinstance(answer, float):
        return 'Z'

    if 'the correct answer is (A)' in answer:
        return 'A'
    elif 'the correct answer is (B)' in answer:
        return 'B'
    elif 'the correct answer is (C)' in answer:
        return 'C'
    elif 'the correct answer is (D)' in answer:
        return 'D'
    
    if answer is None or pd.isna(answer) or '(A)' in answer and '(B)' in answer:
        return "Z"
    if 'Answer: A' in answer or 'A:' in answer or 'correct answer is A' in answer or answer == ' A' or answer == 'A' or '(A)' in answer or answer == ' 1' or answer == '1' or answer == ' 0' or answer == '0':
        return 'A'
    elif 'Answer: B' in answer or 'B:' in answer or 'correct answer is B' in answer or answer == ' B' or answer == 'B' or '(B)' in answer or answer == ' 2' or answer == '2':
        return 'B'
    elif 'Answer: C' in answer or 'C:' in answer or 'correct answer is C' in answer or answer == ' C' or answer == 'C' or '(C)' in answer or answer == ' 3' or answer == '3':
        return 'C'
    elif 'Answer: D' in answer or 'D:' in answer or 'correct answer is D' in answer or answer == ' D' or answer == 'D' or '(D)' in answer or answer == ' 4' or answer == '4':
        return 'D'
    else:
        print(answer)
        return "Z"
        
def preprocess_dataframe(df):
    df['raw_pred'].fillna("No answer", inplace=True)
    df['raw_pred'] = [x.replace("\\n", "") for x in df['raw_pred']]
    df['Explanation for correct answer'].fillna("", inplace=True)
    df['Explanation for correct answer'] = [x.replace("Explanation: ", "") for x in df['Explanation for correct answer']]
    df['raw_pred'] = [x.replace("Explanation: ", "") for x in df['raw_pred']]
    df['raw_pred'] = [x.strip() for x in df['raw_pred']]
    df['raw_pred'] = [x[:x.find("\n") + 1] for x in df['raw_pred']]

    raw_preds = []
    true_preds = []
    for idx, row in enumerate(df.iterrows()):
        sentences_raw = sent_tokenize(row[1]['raw_pred'])
        # print(sentences_raw)
        sentences_true = sent_tokenize(row[1]['Explanation for correct answer'])
        if len(sentences_raw) == 0:
            sentences_raw = [""]
        if len(sentences_true) == 0:
            sentences_true = [""]
        sentences_raw = [x.replace(".", "") for x in sentences_raw]
        sentences_raw = [x.replace("(A", "(A)") for x in sentences_raw]
        sentences_raw = [x.replace("(B", "(B)") for x in sentences_raw]
        sentences_raw = [x.replace("(C", "(C)") for x in sentences_raw]
        sentences_raw = [x.replace("(D", "(D)") for x in sentences_raw]
        sentences_raw = [x.replace("))", ")") for x in sentences_raw]
        sentences_true = [x.replace(".", "") for x in sentences_true]
        # print(row[1]['raw_pred'])
        # print('##########################')

        if '(A)' in sentences_raw[0] or '(B)' in sentences_raw[0] or '(C)' in sentences_raw[0] or '(D)' in sentences_raw[0]:
            sentences_raw = sentences_raw[1: ] + [sentences_raw[0]]

        if '(A)' not in sentences_raw[-1] and '(B)' not in sentences_raw[-1] and '(C)' not in sentences_raw[-1] and '(D)' not in sentences_raw[-1]:
            pass
            # print(idx)
            # print('. '.join(sentences_raw))
        try:
            if '(A)' in sentences_true[0] or '(B)' in sentences_true[0] or '(C)' in sentences_true[0] or '(D)' in sentences_true[0]:
                sentences_true = sentences_true[1: ] + [sentences_true[0]]
        except:
            pass
        try:
            if '(A)' not in sentences_true[-1] and '(B)' not in sentences_true[-1] and '(C)' not in sentences_true[-1] and '(D)' not in sentences_true[-1]:
                sentences_true = sentences_true + [f"Therefore, the correct answer is ({row[1]['correct_answer_letter']})"]
        except:
            pass
        raw_preds.append('. '.join(sentences_raw))
        true_preds.append('. '.join(sentences_true))
    df['raw_pred'] = raw_preds
    df['Explanation for correct answer'] = true_preds
    return df  

def load_model(model_name):
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, attn_implementation='flash_attention_2', device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name)
    return model

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def generate_embedding(sentence, model):
    if isinstance(sentence, float):
        sentence = ""
    with torch.no_grad():
        # batch_dict = tokenizer(sentence, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        # outputs = model(**batch_dict, output_hidden_states=True)
        # embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
        embeddings = torch.Tensor(model.encode(sentence))
    return embeddings

def generate_embeddings_dataframe(df, model):
    embedding_dict = dict([])
    for idx, row in enumerate(df.iterrows()):
        embedding_pred = generate_embedding(row[1]['raw_pred'], model)
        embedding_true = generate_embedding(row[1]['Explanation for correct answer'], model)
        embedding_dict[idx] = (embedding_pred, embedding_true)
    return embedding_dict

def calculate_cosine_similarity(df, embedding_dict):
    similarity = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for idx, row in enumerate(df.iterrows()):
        similarity.append(cos(embedding_dict[idx][0], embedding_dict[idx][1]).item())
    df['cosine_similarity'] = similarity
    return df

def calculate_bert_score(df):
    bert_score = []
    for idx, row in enumerate(df.iterrows()):
        P, R, F1 = score([row[1]['raw_pred']], [row[1]['Explanation for correct answer']], rescale_with_baseline=True, lang="en")
        bert_score.append(max(F1[0].item(), 0))
        # print(max(F1[0].item(), 0))
    df['bert_score'] = bert_score
    return df   

def calculate_rouge_score(df):
    rouge_score = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for idx, row in enumerate(df.iterrows()):
        scores = scorer.score(row[1]['raw_pred'], row[1]['Explanation for correct answer'])
        # print(row[1]['raw_pred'])
        # print(row[1]['Explanation for correct answer'])
        # print(scores['rougeL'].fmeasure)
        # print('@@@@@@@@@@@@@@@@')
        rouge_score.append(scores['rougeL'].fmeasure)
    df['rouge_score'] = rouge_score
    return df    

def calculate_length_difference(df):
    length_difference = []
    length_pred = []
    for idx, row in enumerate(df.iterrows()):
        e1 = len(row[1]['raw_pred']) / 4
        e2 = len(row[1]['Explanation for correct answer']) / 4
        length_difference.append(e1 - e2)
        length_pred.append(e1)
    df['length_difference'] = length_difference
    df['length_prediction'] = length_pred
    return df

def load_prompt(prompt_file):
    prompt = ""
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

def prompt_type(question_data):
    case_1_string = f"Rationale 2 is more comprehensive, factual and coherent compared to Rationale 1"
    case_2_string = f"Rationale 2 is less comprehensive, factual and coherent compared to Rationale 1"
    case_3_string = f"Rationale 2 is equally comprehensive, factual and coherent compared to Rationale 1"
    output_structure_string = f"Your answer should first provide all reasoning followed by the chosen option string in the last line. The chosen option string should exactly match with one of the given options."
    prompt = load_prompt('../prompt_template/correctness.txt')
    prompt = prompt.format(question_data['demographic'], question_data['presenting_problem'], question_data['mental_exam'], question_data['context'], question_data['question'], question_data['answer'], question_data['true_explain'], question_data['pred_explain'], case_1_string, case_2_string, case_3_string, output_structure_string)
    return prompt


def get_eval_correctness(data_df):
    case_1_string = f"Rationale 2 is more comprehensive, factual and coherent compared to Rationale 1"
    case_2_string = f"Rationale 2 is less comprehensive, factual and coherent compared to Rationale 1"
    case_3_string = f"Rationale 2 is equally comprehensive, factual and coherent compared to Rationale 1"
    
    llm_answer_eval_list = []
    case_label_list = []

    
    option_string_list = [case_1_string, case_2_string, case_3_string]

    client = openai_setup()[0]
    for idx, row in enumerate(data_df.iterrows()):
        time.sleep(0.5)
        retry = True
        
        # if idx%1 == 0:
            # print("Index: ", idx)

        while retry:
            try:
                #select one column between answer and question_answered randomly
                message_list = [{"role": "system", "content": "You are a helpful, respectful, honest, and knowledgeable expert mental health counselor"}]
                messages = message_list.copy()
                prompt = prompt_type({'demographic': row[1]['Patient Demographic'], 'presenting_problem': row[1]['Presenting Problem'], 'mental_exam': row[1]['Mental Status Exam'], 'context': row[1]['Other Contexts'], 'question': row[1]['Question'], 'answer': row[1]['Answers'], 'true_explain': row[1]['Explanation for correct answer'], 'pred_explain': row[1]['raw_pred']})
                messages.append({'role': 'user', 'content': prompt})
                
                llm_response = get_response(client, messages, 'counselingqa-gpt4o', 0.1, 1024)
                case_label = ""
                
                if llm_response == "":
                    case_label = "No Response"
                
                else:
                
                    #check if any of the string present in the option_string_list is present in the response if yes then assign that string to case_label_English
                    # llm_response_check_list = (" ".join(llm_response.split("\n"))).split(".")
                    # get the last two sentences from the response
                    # llm_response_check = " ".join(llm_response_check_list[-3:])
                    llm_response_check = llm_response
                    
                    for option_string in option_string_list:
                        if option_string in llm_response_check:
                            case_label = option_string
                            break
                    
                    if case_label == "":
                        case_label = "No Response"
            

                # print("LLM Response: ", llm_response)
                # print("Case Label: ", case_label)

                llm_answer_eval_list.append(llm_response)
                case_label_list.append(case_label)
                #code for saving intermediate files

                retry = False

            except Exception as e:
                print("Error at index: ", idx)
                traceback.print_exc()
                print("Error: ", e)
                #check if the error contains the substring Request timed out: HTTPSConnectionPool or rate limit

                if "Request timed out: HTTPSConnectionPool" in str(e) or "rate limit" in str(e) or "timed out" or "No Response" in str(e):
                    print("Sleeping for 10 seconds")
                    time.sleep(10)
                    continue

                else:
                    #check if llm_response_English exists and is not empty
                    if llm_response:
                        llm_answer_eval_list.append(llm_response)
                        case_label_list.append(case_label)
                    else:
                        if "This model's maximum context length is 8192 tokens" in str(e):
                            llm_answer_eval_list.append("Max Context Length Exceeded")
                            case_label_list.append("Max Context Length Exceeded")

                        else:
                            llm_answer_eval_list.append(str(e))
                            case_label_list.append(str(e))

                    # print("LLM Response: ", llm_response)
                    # print("Case Label: ", case_label)
                    retry = False
                    continue



    data_df["llm_answer_eval"] = llm_answer_eval_list
    data_df["case_label"] = case_label_list
        
    return data_df

def generate_metrics_df(df, filename):
    model = load_model('sentence-transformers/all-mpnet-base-v2')
    embeddings = generate_embeddings_dataframe(df, model)
    
    if 'cosine_similarity' not in df.columns:
        print('Generating cosine similarity...')
        df = calculate_cosine_similarity(df, embeddings)
        df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    if 'bert_score' not in df.columns:
        print('Generating bert score...')
        df = calculate_bert_score(df)
        df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    if 'rouge_score' not in df.columns:
        print('Generating rouge score...')
        df = calculate_rouge_score(df)
        df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    torch.cuda.empty_cache()
    if 'length_difference' not in df.columns:
        print('Generating length difference...')
        df = calculate_length_difference(df)
        df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    # if 'case_label' not in df.columns:
    #     df = get_eval_correctness(df)
    #     df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    return df

def write_metrics(df, filename):
    df.to_csv(f'../results/outputs_reasoning/{filename}.csv', index=False)
    with open(f'../results/metrics_reasoning/{filename}.txt', 'w') as f:
        f.write(f"Average cosine similarity: {np.mean(df['cosine_similarity'])}\n")
        f.write(f"########################################################\n")            
        f.write(f"Average BertScore: {np.mean(df['bert_score'])}\n")
        f.write(f"########################################################\n")        
        f.write(f"Average length difference: {np.mean(df['length_difference'])}\n")
        f.write(f"########################################################\n")
        dd = dict(Counter(df['case_label']))
        for key in dd.keys():
            f.write(str(key) + ": ")
            f.write(str(dd[key]))
            f.write("\n")
            f.write(f"########################################################\n")
        print(f"Saving to ../results/metrics_reasoning/{filename}.txt")
        f.close() 

if __name__ == "__main__":
    for file in os.listdir('../results/outputs'):
        if 'few-shot-cot_None_3' in file:
            print(file)
            filename = file[:file.find("_")]
            # if '7b' in str.lower(filename):
            #     continue
            if not os.path.isfile(f'../results/outputs_reasoning/{filename}.csv'):
                df = pd.read_csv(os.path.join('../results/outputs', file))
                df = preprocess_dataframe(df)
                df = generate_metrics_df(df, filename)
            else:
                try:
                    # if filename == 'counselingQA-gpt4o' or filename == 'BioMedGPT-LM-7B':
                    print(file)
                    df = pd.read_csv(f'../results/outputs_reasoning{filename}.csv')     
                    df = preprocess_dataframe(df)
                    df = generate_metrics_df(df, filename)
                    # write_metrics(df, filename)
                except:
                    continue
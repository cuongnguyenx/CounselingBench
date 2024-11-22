import torch
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from random import shuffle
from collections import Counter
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import os
import datasets
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import time
import pickle
import gc

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used
# print(f'CUDA device count: {torch.cuda.device_count()}')
# print(f'CUDA device name: {torch.cuda.get_device_name("cuda:0")}')
# print(f'CUDA device name: {torch.cuda.get_device_name("cuda:1")}')
# print(f'CUDA device name: {torch.cuda.get_device_name("cuda:2")}')
# print(f'CUDA device name: {torch.cuda.get_device_name("cuda:3")}')

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = "bfloat16", attn_implementation='flash_attention_2', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype="bfloat16", 
        device_map="auto"
    )
    return pipe

def load_vllm(model_name):
    print("Loading model with vllm")
    pipe = LLM(model=model_name, 
               dtype="bfloat16",
               gpu_memory_utilization=0.85,
               tensor_parallel_size=8,
               max_context_len_to_capture=6144,
               )
    print("Model initialized!")
    return pipe

def load_client(model):
    if 'gpt' in model:
        # return OpenAI(api_key="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")
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-02-01",
            azure_endpoint = "https://counselingqa-openaikey.openai.azure.com",
        )
        return client
    elif 'claude' in model:
        return Anthropic(api_key="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")
    elif 'gemini'in model:
        return genai.GenerativeModel(api_key="")
    else:
        return None

def load_prompt(prompt_file):
    prompt = ""
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

def load_dataset(data_file):
    df = pd.read_csv(data_file)
    # df['Correct Answer'] = [x.replace('Correct answer: ', '') for x in df['Correct Answer']]
    # df['Choice A'] = [x[x.find('.') + 1:] for x in df['Choice A']]
    # df['Choice B'] = [x[x.find('.') + 1:] for x in df['Choice B']]
    # df['Choice C'] = [x[x.find('.') + 1:] for x in df['Choice C']]
    # df['Choice D'] = [x[x.find('.') + 1:] for x in df['Choice D']]
    answers = []
    correct_answers = []
    for idx, row in enumerate(df.iterrows()):
        choices = [row[1]['Choice A'], row[1]['Choice B'], row[1]['Choice C'], row[1]['Choice D']]
        # shuffle(choices)
        if row[1]['Correct Answer'] == choices[0]:
            correct_answers.append('A')
        elif row[1]['Correct Answer'] == choices[1]:
            correct_answers.append('B')
        elif row[1]['Correct Answer'] == choices[2]:
            correct_answers.append('C')
        elif row[1]['Correct Answer'] == choices[3]:
            correct_answers.append('D')
        else:
            print(idx)
            print(row[1]['Correct Answer'])
            print(choices)
        answers.append(f"(A): {choices[0]}\n(B): {choices[1]}\n(C): {choices[2]}\n(D): {choices[3]}")

    df['Answers'] = answers
    df['correct_answer_letter'] = correct_answers
    df.to_csv('train.csv')
    file_dict = {
        "train" : "train.csv",
    }

    dataset = datasets.load_dataset(
        'csv',
        data_files=file_dict,
        delimiter=',',
        column_names=list(df.columns),
        split="train"
    )
    return df, dataset

def run_prompt(question_data, mode, pipe, temperature, max_tokens, model_name):
    # print('Running prompt')
    TIRAMISU = True
    global NUM_SHOTS
    res = ""
    res_l = ""
    prompt = ""

    # Load prompt
    if mode == 'zero-shot':
        prompt = load_prompt('../prompt_template/zero_shot_template.txt')
        prompt = prompt.format(question_data['demographic'][0], question_data['presenting_problem'][0], question_data['mental_exam'][0], question_data['context'][0], question_data['question'][0], question_data['answer'][0])
    elif mode == 'zero-shot-cot':
        prompt = load_prompt('../prompt_template/zero_shot_template_cot.txt')
        prompt = prompt.format(question_data['demographic'][0], question_data['presenting_problem'][0], question_data['mental_exam'][0], question_data['context'][0], question_data['question'][0], question_data['answer'][0])
    elif mode == 'few-shot':
        prompt = load_prompt('../prompt_template/few_shot_template.txt')
        tuples = [(question_data['demographic'][i], question_data['presenting_problem'][i], question_data['mental_exam'][i], question_data['context'][i], question_data['question'][i], question_data['answer'][i], question_data['correct_answer'][i]) for i in range(NUM_SHOTS + 1)]
        ll = [item for t in tuples for item in t]
        prompt = prompt.format(*ll)
    elif mode == 'few-shot-cot':
        prompt = load_prompt('../prompt_template/few_shot_template_cot.txt')
        tuples = [(question_data['demographic'][i], question_data['presenting_problem'][i], question_data['mental_exam'][i], question_data['context'][i], question_data['question'][i], question_data['answer'][i], question_data['explanations'][i], question_data['correct_answer'][i]) for i in range(NUM_SHOTS + 1)]
        ll = [item for t in tuples for item in t]
        prompt = prompt.format(*ll)
    
    # print(prompt)
    # print(len(pipe.tokenizer(prompt)['input_ids']))
    
    if ('workingdir' in model_name or TIRAMISU) and 'gpt4o' not in model_name:
        # output = pipe(
        #     prompt,
        #     do_sample=True,
        #     max_new_tokens=max_tokens, 
        #     num_return_sequences=10,
        #     temperature=temperature,
        #     top_p = 0.95,
        #     top_k = 50
        # )
        # res = output[0]['generated_text'][len(prompt):]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
        output = pipe.generate([prompt], sampling_params)
        res =  output[0].outputs[0].text

        res_l = parse_answers(res, 'cot' in mode)
    elif 'gpt' in model_name:
        time.sleep(0.5)
        ress = pipe.chat.completions.create(
            model=model_name,
            messages = [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        res = ress.choices[0].message.content
        res_l = parse_answers(res, 'cot' in mode)
        
    elif 'claude' in model_name:
        attempts = 0
        while attempts < 5:
            try:
                ress = pipe.messages.create(
                    model=model_name,
                    messages = [{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                res = ress.content[0].text
                res_l = parse_answers(res, 'cot' in mode)
                attempts = 5
            except Exception as e:
                print(e)
                time.sleep(0.5)
                attempts += 1

    return res, res_l


def get_results(question_data, mode, pipe, reasoning, model_name):
    max_tokens = 1
    if '-cot' in mode:
        max_tokens = 500
    if reasoning is None or reasoning == '':
        res, res_l = run_prompt(question_data, mode, pipe, 0.001, max_tokens, model_name)
        print(res)
        print(res_l)
        return res, res_l, None
    # Self-consistency
    elif reasoning == 'sc':
        results = []
        results_l = []
        temps = [0.2, 0.4, 0.6, 0.8, 1]
        for temp in temps:
            res_curr = []
            resl_curr = []
            for i in range(5): 
                res, res_l = run_prompt(question_data, mode, pipe, temp, max_tokens, model_name)
                res_curr.append(res)
                resl_curr.append(res_l)
            results.append(res_curr)
            results_l.append(resl_curr)
        try:
            cl = dict(Counter(results_l[0]))
            cl = {k: v for k, v in cl.items() if k in ['A', 'B', 'C', 'D', 'Z']}
            print(results_l)
            print(Counter(results_l[0]).most_common(1)[0][0])
            return Counter(results_l[0]).most_common(1)[0][0], max(cl.items(), key=lambda x: x[1])[0], results_l
        except Exception as e:
            print(e)
            return '', 'Z', None
        
        
def shot_selection(dataset, num_shots):
    shots = dataset.sample(n=num_shots, random_state=42)
    shots = shots[['Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Correct Answer', 'Answers', 'Explanation for correct answer', 'correct_answer_letter', 'Patient Demographic', 'Presenting Problem', 'Mental Status Exam', 'Other Contexts']]
    return shots

def parse_answers(answer, cot):
    if answer is None or isinstance(answer, float):
        return 'Z'
    
    answer = str.lower(answer)
    if cot:
        answer = answer[answer.rfind("\n") + 1:]
        # print(answer)
        # print('@@@@@@@')
    if 'the correct answer is (a)' in answer and 'the correct answer is (b)' in answer:
        return 'Z'
    if 'the correct answer is (a)' in answer or 'correct answer: (a)' in answer or 'the correct answer here is (a)' in answer or 'answer: (a)' in answer or 'the appropriate answer is (a)' in answer or '(a):' in answer or 'the answer here is (a)' in answer:
        return 'A'
    elif 'the correct answer is (b)' in answer or 'correct answer: (b)' in answer or 'the correct answer here is (b)' in answer or 'answer: (b)' in answer or 'the appropriate answer is (b)' in answer or '(b):' in answer or 'the answer here is (b)' in answer:
        return 'B'
    elif 'the correct answer is (c)' in answer or 'correct answer: (c)' in answer or 'the correct answer here is (c)' in answer or 'answer: (c)' in answer or 'the appropriate answer is (c)' in answer or '(c):' in answer or 'the answer here is (c)' in answer:
        return 'C'
    elif 'the correct answer is (d)' in answer or 'correct answer: (d)' in answer or 'the correct answer here is (d)' in answer or 'answer: (d)' in answer or 'the appropriate answer is (d)' in answer or '(d):' in answer or 'the answer here is (d)' in answer:
        return 'D'
    
    a_loc = answer.find('(a)')
    b_loc = answer.find('(b)')
    c_loc = answer.find('(c)')
    d_loc = answer.find('(d)')
    if a_loc == -1:
        a_loc = 1e+9
    if b_loc == -1:
        b_loc = 1e+9
    if c_loc == -1:
        c_loc = 1e+9
    if d_loc == -1:
        d_loc = 1e+9
        
    if answer is None or pd.isna(answer):
        return "Z"
    if 'answer: A' in answer or 'a:' in answer or 'correct answer is a' in answer or 'correct answer: a' in answer or answer == ' a' or answer == 'a' or answer == ' 1' or answer == '1' or answer == ' 0' or answer == '0':
        return 'A'
    elif 'answer: b' in answer or 'b:' in answer or 'correct answer is b' in answer or 'correct answer: b' in answer  or answer == ' b' or answer == 'b' or answer == ' 2' or answer == '2':
        return 'B'
    elif 'answer: c' in answer or 'c:' in answer or 'correct answer is c' in answer or 'correct answer: c' in answer  or answer == ' c' or answer == 'c' or answer == ' 3' or answer == '3':
        return 'C'
    elif 'answer: d' in answer or 'd:' in answer or 'correct answer is d' in answer or 'correct answer: d' in answer  or answer == ' d' or answer == 'd' or answer == ' 4' or answer == '4':
        return 'D'
    
    if min(a_loc, b_loc, c_loc, d_loc) == a_loc:
        return 'A'
    if min(a_loc, b_loc, c_loc, d_loc) == b_loc:
        return 'B'
    if min(a_loc, b_loc, c_loc, d_loc) == c_loc:
        return 'C'
    if min(a_loc, b_loc, c_loc, d_loc) == d_loc:
        return 'D'
    
    return 'Z'

def benchmark_qa(model_name, dataset_dir, mode, reasoning, setting_str):
    TIRAMISU = True
    if os.path.isfile(f'../results/outputs/{setting_str}.csv'):
        print('Issa Done')
        return 0
    if ('workingdir' in model_name or TIRAMISU) and 'gpt4o' not in model_name:
        # model, tokenizer = load_model(model_name)
        # pipeline = load_pipeline(model, tokenizer)
        pipeline = load_vllm(model_name)
    else:
        pipeline = load_client(model_name)
    df, dataset = load_dataset(dataset_dir)
    shots = None
    if mode == 'few-shot' or mode == 'few-shot-cot':
        shots = shot_selection(df, NUM_SHOTS)
    
    results = []
    results_l = []
    results_sc = []

    # Trying to initialize result arrays from temp files
    if os.path.exists(f"./temp/{setting_str}.pkl"):
        with open(f"./temp/{setting_str}.pkl", "rb") as f:
            results_comb = pickle.load(f)
            results = results_comb[0]
            results_l = results_comb[1]
            results_sc = results_comb[2]
            print(len(results))
            print(results)
            print(results_l)

    for idx, row in enumerate(df.iterrows()):
        print(idx)
        if idx < len(results_l):
            print("Skipping")
            continue
        if idx % 2 == 0 and idx > 0:
            with open(f"./temp/{setting_str}.pkl", "wb") as f:
                pickle.dump([results, results_l, results_sc], f)
        context = row[1]['Other Contexts']
        demo = row[1]['Patient Demographic']
        problem = row[1]['Presenting Problem']
        mental_exam = row[1]['Mental Status Exam']
        # context = ''
        question = row[1]['Question']
        answer = row[1]['Answers']
        res = None
        res_l = None
        res_sc = None
        if mode == 'zero-shot' or mode == 'zero-shot-cot':
            res, res_l, res_sc = get_results({'context': [context], 'question': [question], 'answer': [answer], 'demographic': [demo], 'presenting_problem': [problem], 'mental_exam': [mental_exam]}, mode, pipeline, reasoning, model_name)
            results.append(res)
            results_l.append(res_l)
        elif mode == 'few-shot':
            contexts = ['' for i in range(len(shots['Other Contexts']))] + [context]
            questions = list(shots['Question']) + [question]
            p_answers = list(shots['Answers']) + [answer]
            demos = ['' for i in range(len(shots['Patient Demographic']))] + [demo]
            problems = ['' for i in range(len(shots['Presenting Problem']))] + [problem]
            mental_exams = ['' for i in range(len(shots['Mental Status Exam']))] + [mental_exam]
            t_answers = list(shots['correct_answer_letter']) + ['']
            res, res_l, res_sc = get_results({'context': contexts, 'demographic': demos, 'presenting_problem': problems, 'mental_exam': mental_exams, 'question': questions, 'answer': p_answers, 'correct_answer': t_answers}, mode, pipeline, reasoning, model_name)
            results.append(res)
            results_l.append(res_l)
            # print(shots)
        elif mode == 'few-shot-cot':
            contexts = ['' for i in range(len(shots['Other Contexts']))] + [context]
            questions = list(shots['Question']) + [question]
            p_answers = list(shots['Answers']) + [answer]
            demos = ['' for i in range(len(shots['Patient Demographic']))] + [demo]
            problems = ['' for i in range(len(shots['Presenting Problem']))] + [problem]
            mental_exams = ['' for i in range(len(shots['Mental Status Exam']))] + [mental_exam]
            exps = list(shots['Explanation for correct answer']) + ['']
            t_answers = list(shots['correct_answer_letter']) + ['']
            res, res_l, res_sc = get_results({'context': contexts, 'demographic': demos, 'presenting_problem': problems, 'mental_exam': mental_exams, 'question': questions, 'answer': p_answers, 'explanations': exps, 'correct_answer': t_answers}, mode, pipeline, reasoning, model_name)
            results.append(res)
            results_l.append(res_l)
            # print(shots)
        if reasoning == 'sc':
            results_sc.append(res_sc)
    
    if reasoning == 'sc':
        df['consistency'] = results_sc
    df['raw_pred'] = results
    df['parsed_pred'] = results_l
    df.to_csv(f'../results/outputs/{setting_str}.csv')
    # Removing temp results file after full results are written
    os.remove(f"./temp/{setting_str}.pkl")
    
    metrics = generate_metrics(results_l, list(df['correct_answer_letter']))
    print(metrics)
    with open(f'../results/metrics/{setting_str}.txt', 'w') as f:
        for key in metrics:
            f.write(key)
            f.write("\n")
            f.write(str(metrics[key]))
            f.write("\n")
        print(f"Saving to ../results/metrics/{setting_str}.txt")
        f.close() 
        
    # Clear model from GPU memory
    destroy_model_parallel()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    
    return metrics

def map_output(arr):
    # Assuming arr is an array consisting of 'A', 'B', 'C', and 'D'
    new_arr = []
    for elem in arr:
        if elem == 'A':
            new_arr.append(0)
        elif elem == 'B':
            new_arr.append(1)
        elif elem == 'C':
            new_arr.append(2)
        elif elem == 'D':
            new_arr.append(3)
    return new_arr

def generate_metrics(pred_answers, correct_answers):
    # Accuracy
    print(pred_answers)
    print(correct_answers)
    p_ans = map_output(pred_answers)
    c_ans = map_output(correct_answers)
    acc = accuracy_score(c_ans, p_ans)
    f1 = f1_score(c_ans, p_ans, average='macro')
    precision = precision_score(c_ans, p_ans, average='macro')
    recall = recall_score(c_ans, p_ans, average='macro')
    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, help='Model ID on Huggingface')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Location of QA dataset')
    parser.add_argument('--mode', dest='mode', type=str, help='Training strategy deployed')
    parser.add_argument('--numshots', dest='numshots', type=str, default=0, help='Number of examples used for few-shot learning')
    parser.add_argument('--reasoning', dest='reasoning', type=str, default=None, help='Type of reasoning used')
    args = parser.parse_args()
    model_name = args.model_name[args.model_name.rfind('/') + 1:]
    setting_str = f'{model_name}_{args.mode}_{args.reasoning}_{args.numshots}'
    NUM_SHOTS = int(args.numshots)
    metrics = benchmark_qa(args.model_name, args.dataset, args.mode, args.reasoning, setting_str)
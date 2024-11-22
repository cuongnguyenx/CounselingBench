import os
import sys
import subprocess
# 
# models = ['Llama3-Med42-70B', 'Asclepius-7B', 'Asclepius-13B', 'Asclepius-Llama3-8B', 'Meta-Llama-3.1-8B', 'Meta-Llama-3.1-70B', 'Llama3-OpenBioLLM-8B', 'Llama3-OpenBioLLM-70B',  'ClinicalCamel-70B', 'Llama3-Med42-70B', 'Llama3-Med42-8B', 'meditron-70b', 'meditron-7b', 'Mistral-7B-Instruct-v0.3', 'Meta-Llama-3-8B-Instruct', 'Llama-2-7b-hf', 'llama-2-7b-chat-hf', 'Llama-2-13b-hf', 'Llama-2-13b-chat-hf', 'MentaLLaMA-chat-13B', 'BioMedGPT-LM-7B', 'medalpaca-7b', 'medalpaca-13b', 'Meta-Llama-3-70B-Instruct', 'Llama-2-70b-chat-hf', 'med42-70b', 'Llama-2-70b-hf', 'counselingQA-gpt4o']
models = ['m42-health/Llama3-Med42-70B', 'starmpcc/Asclepius-7B', 'starmpcc/Asclepius-13B', 'starmpcc/Asclepius-Llama3-8B', 'aaditya/Llama3-OpenBioLLM-8B', 'aaditya/Llama3-OpenBioLLM-70B',  'wanglab/ClinicalCamel-70B', 'm42-health/Llama3-Med42-8B', 'epfl-llm/meditron-70b', 'epfl-llm/meditron-7b', 'mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-hf', 'meta-llama/llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf', 'klyang/MentaLLaMA-chat-13B', 'PharMolix/BioMedGPT-LM-7B', 'medalpaca/medalpaca-7b', 'medalpaca/medalpaca-13b', 'meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Llama-2-70b-chat-hf', 'm42-health/med42-70b', 'meta-llama/Llama-2-70b-hf']
modes = ['zero-shot', 'few-shot']
reasonings = ['sc']
TIRAMISU = True
for model in models:
    for mode in modes:
        for reasoning in reasonings:
            print(f"{model} -- {mode} -- {reasoning}")
            if 'gpt' in model or 'claude' in model or TIRAMISU:
                command_1 = f"python3 qa_benchmark_llamalike.py --model_name {model} --dataset ../data/mct_combined.csv --mode {mode}" 
            else:
                command_1 = f"python3 qa_benchmark_llamalike.py --model_name ../../../../../workingdir/cnguyen319/{model} --dataset ../data/counselingbench.csv --mode {mode}"
            if reasoning != '':
                command_1 += f" --reasoning {reasoning}"
            if mode == "few-shot" or mode == 'few-shot-cot':
                command_1 +=  f" --numshots 3"
            subprocess.run(command_1, shell=True)
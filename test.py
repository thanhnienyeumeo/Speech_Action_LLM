import datasets
from datasets import load_dataset
import json

dataset = load_dataset('Colder203/Robot_Interaction')
for i in range(len(dataset['train'])):
    if '. ' in dataset['train'][i]['command']:
        dataset['train'][i]['command'] = dataset['train'][i]['command'].split('. ')[1]
        print('spotted')
print('----------')
for i in range(len(dataset['train'])):
    if '. ' in dataset['train'][i]['command']:
        # dataset['train'][i]['command'] = dataset['train'][i]['command'].split('. ')[1]
        print('Spotted')
exit()
#save again to jsonl
dataset.save_to_disk('output_3_5_500_2.jsonl')
name_hub = 'Colder203/Robot_Interaction'
dataset.push_to_hub(name_hub)

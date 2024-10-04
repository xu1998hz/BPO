import json

for i in range(200, 1401, 200):
    data = json.load(open(f'validation_results/sft_{i}.json'))
    print(i)
    print(len(data['scores']))
    print(sum(data['scores'])/len(data['scores']))
    print()
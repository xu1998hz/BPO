import json

final_data = {}

for iter_index in range(8):
    start_index = list(range(0, 2001, 250))[iter_index]
    end_index = list(range(0, 2001, 250))[iter_index+1]
    data = json.load(open(f'data/sample_{iter_index}_{start_index}_{end_index}.json'))
    for key, val in data.items():
        final_data[key] = val

with open(f'sample_10000_12000.json', 'w') as f:
    json.dump(final_data, f)
    print(len(final_data))
    print("File is saved!")
import json

with open('../output/test.json') as f:
    test_data = json.load(f)

with open('../output/test_modelip.txt', 'w') as f:
    for respons in test_data:
        for res in respons:
            if res["entity_group"] == 'yes':
                f.write('1.0')
            else:
                f.write('0.0')


with open('../output/test_modelip.txt') as f:
    output = f.readlines()

with open('../output/answer.txt', 'w') as f:
    for i in range(1, 1581):
        f.write('neutral\n')
    for i in range(1581, 9271):
        f.write('0.0\n')
    for line in output:
        f.write(line)


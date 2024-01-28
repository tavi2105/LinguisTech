with open('../output/test_modelip.txt') as f:
    output = f.readlines()

with open('../output/test_modelip_4e.txt') as f:
    output_ = f.readlines()

with open('../output/answer.txt', 'w') as f:
    for i in range(1, 1581):
        f.write('neutral\n')
    for i in range(1581, 9271):
        f.write('0.0\n')
    for line1, line2 in zip(output, output_):
        if line1 == "1.0\n" and line2 == "1.0\n":
            f.write("1.0\n")
        else:
            f.write("0.0\n")


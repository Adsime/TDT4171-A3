import functions as f

test = "./data/test.txt"
train = "./data/training.txt"

test_examples = f.file_to_matrix(test)
train_examples = f.file_to_matrix(train)
attributes = [0, 1, 2, 3, 4, 5, 6]

f.test_random(train_examples, test_examples, attributes, 100000)
f.test_gain(train_examples, test_examples, attributes)
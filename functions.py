import random
import math
import numpy as np
import tree
import time
import os


def file_to_matrix(file):
    f = open(file)
    lines = []
    for line in f.readlines():
        lines.append(line.replace("\n", "").split("\t"))
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            lines[i][j] = 1 if c == '1' else 2
    return lines


def importance_random():
    return random.uniform(0, 1)


def b_func(q):
    return -((q * math.log(q, 2)) + ((1 - q) * math.log((1 - q), 2))) if 0 < q < 1 else 0


def pos_count(examples):
    p = 0
    for example in examples:
        p += 1 if example[-1] == 2 else 0
    return p


def importance_gain(attribute, examples):
    e = [[] for i in range(2)]
    for example in examples:
        if example[attribute] == 1:
            e[0].append(example)
        else:
            e[1].append(example)
    p = pos_count(examples)
    gain = b_func(p / len(examples)) - remainder(e)
    return gain


def remainder(e):
    rem = 0
    for arr in e:
        rem += (len(arr) / len(e)) * b_func(pos_count(arr) / len(arr)) if len(arr) != 0 else 0
    return rem


def plurality_values(examples):
    p = pos_count(examples)
    return random.choice([1, 2]) if p == (len(examples)/2) else 2 if p > (p - len(examples)) else 1


def decision_tree_learning(examples, attributes, parent_examples, importance_method):
    if len(examples) == 0:
        return tree.Tree(plurality_values(parent_examples))
    elif all(examples[0][-1] == example[-1] for example in examples):
        return tree.Tree(examples[0][-1])
    elif len(attributes) == 0:
        return tree.Tree(plurality_values(examples))
    else:
        weights = []
        for i in attributes:
            weights.append(importance_gain(i, examples) if importance_method == 1 else importance_random())
        A = np.argmax(weights)
        t = tree.Tree(attr=attributes[A])
        for a in [1, 2]:
            exs = []
            for e in examples:
                if e[attributes[A]] == a:
                    exs.append(e)
            newAttr = list(attributes)
            newAttr.remove(attributes[A])
            branch = decision_tree_learning(exs, newAttr, examples, importance_method)
            t.append_branch(branch)
        return t


def test_random(train_examples, test_examples, attributes, iterations):
    holder = 0
    low, high = [1, 0]
    low_tree, high_tree = [None, None]
    time_m = time.time()
    for i in range(iterations):
        if (i+1) % (iterations/100) == 0:
            os.system('cls')
            seconds = (time.time() - time_m)
            mins = math.floor(seconds/60)
            seconds = seconds - (mins * 60)
            print("Progress: " + ((i+1)*100/iterations).__str__() + "%. Time taken so far: " +
                  mins.__str__() + " min and " + seconds.__str__() + " seconds!\n\n")
        c = 0
        t = decision_tree_learning(train_examples, attributes, [], 0)
        for line in test_examples:
            c += 1 if t.classify(line) == line[-1] else 0
        c = c / len(test_examples)
        holder += c
        if low > c:
            low = c
            low_tree = t
        if high < c or high_tree and high == c and high_tree.count_nodes() > t.count_nodes():
            high = c
            high_tree = t

    print(iterations.__str__() + " iterations classification success rate: " + (holder / iterations).__str__())
    print(iterations.__str__() + " iterations success interval: [" + low.__str__() + ", " + high.__str__() + "]")
    print("\nWorst performing tree:\n" + low_tree.to_string(0) + "\n")
    print("Best performing tree:\n" + high_tree.to_string(0) + "\n\n")


def test_gain(train_examples, test_examples, attributes):
    t = decision_tree_learning(train_examples, attributes, [], 1)
    correct = 0
    for line in test_examples:
        correct += 1 if t.classify(line) == line[-1] else 0
    print("Gain tree:\n" + t.to_string(0) + "\n")
    print("Successfull classification rate: " + (correct / len(test_examples)).__str__())
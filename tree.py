class Tree:
    default = "Choice"

    def __init__(self, name=default, attr=-1):
        self.name = name
        self.attr = attr
        self.branches = []
        pass

    def classify(self, example):
        return self.branches[example[self.attr] - 1].classify(example) if self.name == self.default else self.name

    def append_branch(self, children):
        self.branches.append(children)

    def to_string(self, indents):
        text = ""
        for i in range(0, indents):
            text += "\t"
        text += ("Leaf: " + self.name.__str__()) if self.attr == -1 \
            else self.name.__str__() + ": " + self.attr.__str__()
        text += "\n"
        indents += 1
        for b in self.branches:
            text += b.to_string(indents)
        return text

    def count_nodes(self):
        counter = 0
        for branch in self.branches:
            counter += branch.count_nodes()
        return 1 + counter

import nltk
from nltk.corpus import dependency_treebank
import numpy as np
import Chu_Liu_Edmonds_algorithm as cle
from collections import defaultdict
from scipy.sparse import csr_matrix

nltk.download("dependency_treebank")


class MST:

    def __init__(self, feature_function):
        self.sents = dependency_treebank.parsed_sents()
        threshold = int(len(self.sents) * 0.9)
        self.train = self.sents[:threshold]
        self.test = self.sents[threshold:]
        self.feature = feature_function

    def update_weights(self, w, mst, s, lr=1):
        for arc in mst.values():
            arc_features = self.feature(arc.tail, arc.head, s)
            for k in arc_features:
                w[k] -= lr * arc_features[k]
        for node in s.nodes.values():
            if node["head"] is not None:
                arc_features = self.feature(node["address"], node["head"], s)
                for k in arc_features:
                    w[k] += lr * arc_features[k]
        return w

    def perceptron(self, num_iterations, lr):
        Theta = []
        Ws = [defaultdict(float)]
        N = num_iterations * len(self.train)
        w_final = defaultdict(float)
        for i in range(num_iterations):
            for j, parsed_sent in enumerate(self.train):
                arcs = self.generate_arcs_from_sent(parsed_sent, Ws[-1])
                mst = cle.min_spanning_arborescence(arcs, 0)
                new_w = self.update_weights(Ws[-1], mst, parsed_sent, lr)
                Ws.append(new_w)
                if j == 500:
                    return w_final
                if j % 100 is 0:
                    print("Parsed %d/%d.." % (j + 1, N))
                for k in new_w:
                    w_final[k] += float(new_w[k]) / N

        print("Finished calculating the perceptron")
        return w_final

    def generate_arcs_from_sent(self, sent, w):
        size = len(sent.nodes)
        arcs = []
        for u_idx in range(size):
            for v_idx in range(size):
                if u_idx == v_idx:
                    continue
                score = self.score(u_idx, v_idx, sent, w)
                arcs.append(cle.Arc(u_idx, score, v_idx))
        return arcs

    def score(self, u, v, s, w):
        f = self.feature(u, v, s)
        return -sum(w[k] * f[k] for k in f)

    def eval(self, w):
        accs = []
        for i, parsed_sent in enumerate(self.test):
            arcs = self.generate_arcs_from_sent(parsed_sent, w)
            mst = cle.min_spanning_arborescence(arcs, 0)
            mst_arcs = set((arc.tail, arc.head) for arc in mst.values())
            sent_arcs = set(
                (node["address"],node["head"]) for node in parsed_sent.nodes.values() if node["head"] is not None)
            accs.append(len(mst_arcs.intersection(sent_arcs)) / len(parsed_sent.nodes))
            if i % 100 is 0:
                print("Evaluated %d/%d.." % (i + 1, len(self.test)))
        return np.average(accs)


def default_feature(u_idx, v_idx, s):
    feature_vector = defaultdict(float)
    u_word, v_word = s.nodes[u_idx]["word"], s.nodes[v_idx]["word"]
    u_tag, v_tag = s.nodes[u_idx]["tag"], s.nodes[v_idx]["tag"]
    if u_idx is 0:
        u_word, u_tag = "ROOT", "ROOT"
    if v_idx is 0:
        v_word, v_tag = "ROOT", "ROOT"
    feature_vector[(u_word, v_word)] = 1
    feature_vector[(u_tag, v_tag)] = 1
    return feature_vector


ONE_DIST = "one word dist"
TWO_DIST = "two words dist"
THREE_DIST = "three words dist"
FOUR_PLUS_DIST = "for or more words dist"


def dist_feature(u_idx, v_idx, s):
    f = default_feature(u_idx, v_idx, s)
    f[ONE_DIST] = int(u_idx - v_idx == 1)
    f[TWO_DIST] = int(u_idx - v_idx == 2)
    f[THREE_DIST] = int(u_idx - v_idx == 3)
    f[FOUR_PLUS_DIST] = int(u_idx - v_idx >= 4)
    return f


def main():
    default_mst = MST(default_feature)
    dist_mst = MST(dist_feature)
    def_w = default_mst.perceptron(1, 1)
    def_acc = default_mst.eval(def_w)
    print(def_acc)
    print("%.2f" % def_acc)
    dist_w = dist_mst.perceptron(1, 1)
    dist_acc = dist_mst.eval(dist_w)
    print(dist_acc)
    print("%.2f" % dist_acc)


if __name__ == "__main__":
    main()

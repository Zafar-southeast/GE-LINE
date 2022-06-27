import numpy as np

from ge.classify import read_node_label, Classifier
from ge import RELINE

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def split_data(networks):
    # This Functiion splits the data into 80% train and 20% Test
    #
    # input : Networks to split (G1, G2, G3, etc.)
    # return: Returns the 80% train and 20% test giles

    return #G_train, G_test, G1_train, G1_test

def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')



if __name__ == "__main__":
    G = nx.read_edgelist('C:\\Users\\zafar\\PycharmProjects\\GE-LINE\\data\\Example\\authors_authors_v12.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])    # We need to define the path
    G1 = nx.read_edgelist('C:\\Users\\zafar\\PycharmProjects\\GE-LINE\\data\\Example\\papers_references_v12.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])  # We need to define the path

   # G_train, G_test, G1_train, G1_test = split_data(G, G1)

    model = RELINE(G, embedding_size=128, order='second')   # Load the model from the file with specific embeding size and proximity type (i.e first/second order)
    model.train(batch_size=1024, epochs=1, verbose=2)       # Train model with specidic batch size and epochs
    embeddings = model.get_embeddings()                     # Get embeddings

    model1 = RELINE(G1, embedding_size=128, order='first')  # Load the model from the file with specific embeding size and proximity type (i.e first/second order)
    model1.train(batch_size=1024, epochs=1, verbose=2)      # Train model with specidic batch size and epochs
    embeddings1 = model1.get_embeddings()  # Get embeddings

    #print(model.get_embeddings().keys())
    a = model1.get_embeddings()['192821']  # Node A embeddings (first graph) eg. paper-paper
    b = model.get_embeddings()['2019947963']  # Node B embeddings (sendond graph) eg. Author-author
    c = model1.get_embeddings()['1997295686']    #paper-paper
    #print(np.dot(a, b))
    score=np.dot(a, b)
    print(score)
    #final_score=np.dot(two_dot, c)



#for n1, n2 in list(G.edges(data=False)):
    #print(n1, n2) # print the nodes IDs
    #print(a, b)     # print the embeddings for 75 (author ID) and 67 (paper ID) nodes
    #print(model.get_embeddings()[n1])

    #evaluate_embeddings(embeddings) # Evaluate embedings
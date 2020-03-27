import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
from graphviz import render

# download https://graphviz.gitlab.io/_pages/Download/Download_windows.html
os.environ["PATH"] += os.pathsep + os.path.abspath("Graphviz2.38/bin/")


# create state space and initial state probabilities

def init_state_probabilities(states, pi):
    state_space = pd.Series(pi, index=states, name='states')
    return state_space


states = ['sleeping', 'eating', 'playing']
pi = [0.35, 0.35, 0.3]
state_space = init_state_probabilities(states, pi)
print(state_space)
print(state_space.sum())


# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states


def create_transition_matrix(states, transitions):
    length = len(states)
    if length != len(transitions):
        raise Exception("marixs and states length aren't equal")
    q_df = pd.DataFrame(columns=states, index=states)
    for i in range(0, len(states)):
        if length != len(transitions[i]):
            raise Exception("marixs and states length aren't equal at transitions[" + str(i) + "]")
        q_df.loc[states[i]] = transitions[i]
    return q_df


transitions = [[0.4, 0.2, 0.4],
               [0.45, 0.45, 0.1],
               [0.45, 0.25, 0.3]]

q_df = create_transition_matrix(states, transitions)


def print_transition_matrix():
    print(q_df)

    q = q_df.values
    print('\n', q, q.shape, '\n')
    print(q_df.sum(axis=1))


print_transition_matrix()


# create a function that maps transition probability dataframe
# to markov edges and weights


def get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


edges_wts = get_markov_edges(q_df)
pprint(edges_wts)


##############


def create_graph_object(states, edges_wts):
    # create graph object
    G = nx.MultiDiGraph()

    # nodes correspond to states
    G.add_nodes_from(states)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    return G


G = create_graph_object(states, edges_wts)
print(f'Nodes:\n{G.nodes()}\n')
print(f'Edges:')
pprint(G.edges(data=True))


def draw_on_dot(G, filename):
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)

    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.drawing.nx_pydot.write_dot(G, filename)
    render('dot', 'png', filename)


draw_on_dot(G, "pet_dog_markov.dot")

exit()
################################################

# create state space and initial state probabilities

hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())

######################################################

# create hidden transition matrix
# a or alpha
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))

############################################################

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states
# and O is number of different possible observations

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))

###############################################

# create graph edges and weights

hide_edges_wts = get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = get_markov_edges(b_df)
pprint(emit_edges_wts)

##############################################

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
print(f'Nodes:\n{G.nodes()}\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

print(f'Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_hidden_markov.dot')
render('dot', 'png', 'pet_dog_hidden_markov.dot')

##########################################################

# observation sequence of dog's behaviors
# observations are encoded numerically

obs_map = {'sleeping': 0, 'eating': 1, 'playing': 2}
obs = np.array([1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1])

inv_obs_map = dict((v, k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print(pd.DataFrame(np.column_stack([obs, obs_seq]),
                   columns=['Obs_code', 'Obs_seq']))


##############################################

# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    print('-' * 50)
    print('Start Backtrace\n')
    path[T - 1] = np.argmax(delta[:, T - 1])
    # p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T - 2, -1, -1):
        # path[t] = phi[path[t + 1], [t + 1]]
        path[t] = phi[int(path[t + 1]), [t + 1]]
        # p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi


path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

#############################################################

state_map = {0: 'healthy', 1: 'sick'}
state_path = [state_map[v] for v in path]

print(pd.DataFrame()
      .assign(Observation=obs_seq)
      .assign(Best_Path=state_path))

####################################################

import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pylab
from pprint import pprint
from graphviz import render

# download https://graphviz.gitlab.io/_pages/Download/Download_windows.html

os.environ["PATH"] += os.pathsep + os.path.abspath("Graphviz2.38/bin/")


# create state space and initial state probabilities

def init_state_probabilities(states, pi):
    state_space = pd.Series(pi, index=states, name='states')
    return state_space


# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states


def create_transition_matrix(hidden_states, observable_states, transitions):
    length = len(observable_states)
    if len(hidden_states) != len(transitions):
        raise Exception("marixs and states length aren't equal")
    df = pd.DataFrame(columns=observable_states, index=hidden_states)
    for i in range(0, len(hidden_states)):
        if length != len(transitions[i]):
            raise Exception("marixs and states length aren't equal at transitions[" + str(i) + "]")
        df.loc[hidden_states[i]] = transitions[i]
    return df


def print_transition_matrix(q_df):
    print(q_df)

    q = q_df.values
    print('\n', q, q.shape, '\n')
    print(q_df.sum(axis=1))


# create a function that maps transition probability dataframe
# to markov edges and weights


def get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


def create_graph_object(states, edges_wts, emit_edges_wts):
    # create graph object
    G = nx.MultiDiGraph()

    # nodes correspond to states
    G.add_nodes_from(states)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    for k, v in emit_edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    return G



##############################################


def draw_on_dot(G, filename, prog):
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog=prog)
    nx.draw_networkx(G, pos)

    #nx.draw(G, pos, with_labels=True)
    #edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #matplotlib.pyplot.savefig("pyplot."+filename+".png")
    #matplotlib.pyplot.show()

    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.drawing.nx_pydot.write_dot(G, filename)
    render('dot', 'png', filename)


def make_observation(obs, obs_map):
    inv_obs_map = dict((v, k) for k, v in obs_map.items())
    obs_seq = [inv_obs_map[v] for v in list(obs)]

    return obs_seq


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


'''
def show_markov():
    ####
    states = ['sleeping', 'eating', 'playing']
    pi = [0.35, 0.35, 0.3]
    state_space = init_state_probabilities(states, pi)
    print(state_space)
    print(state_space.sum())
    ####
    transitions = [[0.4, 0.2, 0.4],
                   [0.45, 0.45, 0.1],
                   [0.45, 0.25, 0.3]]

    q_df = create_transition_matrix(states, transitions)
    ####
    print_transition_matrix(q_df)
    ####
    edges_wts = get_markov_edges(q_df)
    pprint(edges_wts)
    ####
    G = create_graph_object(states, edges_wts, {})
    print(f'Nodes:\n{G.nodes()}\n')
    print(f'Edges:')
    pprint(G.edges(data=True))
    ####
    draw_on_dot(G, "pet_dog_markov.dot", "dot")
    ####
'''

def show_hidden(states, hidden_states, pi, hidden_transitions, transitions, obs_map, obs, state_map, filename, prog):
    ####
    state_space = init_state_probabilities(hidden_states, pi)
    print(state_space)
    print('\n', state_space.sum())
    ####
    a_df = create_transition_matrix(hidden_states, hidden_states, hidden_transitions)
    print(a_df)
    a = a_df.values
    print('\n', a, a.shape, '\n')
    print(a_df.sum(axis=1))
    ####
    observable_states = states
    b_df = create_transition_matrix(hidden_states, observable_states, transitions)
    print(b_df)
    b = b_df.values
    print('\n', b, b.shape, '\n')
    print(b_df.sum(axis=1))
    ####
    hide_edges_wts = get_markov_edges(a_df)
    print("hide_edges_wts: ")
    pprint(hide_edges_wts)
    emit_edges_wts = get_markov_edges(b_df)
    print("emit_edges_wts: ")
    pprint(emit_edges_wts)
    ####
    G = create_graph_object(hidden_states, hide_edges_wts, emit_edges_wts)
    print(f'Nodes:\n{G.nodes()}\n')
    print(f'Edges:')
    pprint(G.edges(data=True))
    draw_on_dot(G, filename, prog)
    ####
    obs_seq = make_observation(obs, obs_map)
    print(pd.DataFrame(np.column_stack([obs, obs_seq]),
                        columns=['Obs_code', 'Obs_seq']))
    ####
    path, delta, phi = viterbi(pi, a, b, obs)
    print('\nsingle best state path: \n', path)
    print('delta:\n', delta)
    print('phi:\n', phi)
    ####
    state_path = [state_map[v] for v in path]

    print(pd.DataFrame()
          .assign(Observation=obs_seq)
          .assign(Best_Path=state_path))


def show():

    filename = 'student_hidden_markov.dot'
    # prog = 'dot'
    prog = 'neato'
    states = ['sleeping', 'eating', 'playing', 'working out', 'studying']
    hidden_states = ['healthy', 'sick', 'depressed']
    hidden_transitions = [[0.5, 0.2, 0.3],
                          [0.2, 0.5, 0.3],
                          [0.2, 0.3, 0.5]]
    transitions = [[0.2, 0.2, 0.2, 0.2, 0.2],
                   [0.4, 0.3, 0.1, 0.1, 0.1],
                   [0.3, 0.3, 0.3, 0.05, 0.05]]
    hidden_pi = [0.4, 0.3, 0.3]
    obs_map = {'sleeping': 0, 'eating': 1, 'playing': 2, 'working out':3, 'studying':4}
    obs = np.array([3, 3, 2, 3, 0, 1, 2, 3, 2, 4, 4, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    #obs = np.array([1, 2, 3, 3, 1, 4, 4, 4, 1, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 2, 0, 1, 4, 1, 0, 1])
    state_map = {0: 'healthy', 1: 'sick', 2: 'depressed'}
    show_hidden(states, hidden_states, hidden_pi, hidden_transitions, transitions, obs_map, obs, state_map, filename, prog)


show()

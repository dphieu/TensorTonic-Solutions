import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_cnt = np.zeros(n_states)

    for e in episodes:
        G = 0
        for t in reversed(range(len(e))):
            state, reward = e[t]
            G = gamma * G + reward
            if state not in [x[0] for x in e[:t]]:
                returns_sum[state] += G
                returns_cnt[state] += 1
                V[state] = returns_sum[state] / returns_cnt[state]
    return V
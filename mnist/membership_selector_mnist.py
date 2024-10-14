import numpy as np


# MS For imbalance dataset
def membership_selector(membership, eta):
    selected_idx = np.zeros(shape=(10, eta), dtype=np.int32)
    selected_pseudo_label = np.zeros(shape=(10, eta), dtype=np.int32)
    for i in range(10):
        membership_i_class = membership[:, i]
        top_eta_idx = membership_i_class.argsort()[::-1][0:eta]
        selected_idx[i, :] = top_eta_idx
        selected_pseudo_label[i, :] = np.ones(shape=(eta, ), dtype=np.int32) * i
    selected_idx = np.reshape(selected_idx, newshape=(-1, ))
    selected_pseudo_label = np.reshape(selected_pseudo_label, newshape=(-1, ))
    return selected_idx, selected_pseudo_label

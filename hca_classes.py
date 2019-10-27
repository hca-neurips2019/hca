# Classes for the algorithms

class StateHCA(object):
  def __init__(self, n_s, n_a):
    self.h_dim = n_s
    self.n_a = n_a

  def update(self, pi, V, h, states, actions, rewards, gamma):
    T = len(states)
    dlogits = np.zeros_like(pi)
    dV = np.zeros_like(V)
    dlogits_h = np.zeros_like(h)
    
    for i in range(T):
      x_s, a_s = states[i], actions[i]
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_hca = np.zeros(self.n_a)
      
      for j in range(i, T):
        x_t, r = states[j], rewards[j]
        hca_factor = h[:, x_s, x_t].T - pi[x_s, :] 
        G_hca += gamma**(j - i) * r * hca_factor

        dlogits_h[a_s, x_s, x_t] += 1
        dlogits_h[:, x_s, x_t] -= h[:, x_s, x_t]

      for a in range(self.n_a):
        dlogits[x_s, a] += G_hca[a]
        dlogits[x_s] -= pi[x_s] * G_hca[a]
      dV[x_s] += (G - V[x_s])

    return dlogits, dV, dlogits_h

class ReturnHCA(object):
  def __init__(self, n_s, n_a, return_bins):
    self.h_dim = len(return_bins)
    self._return_bins = return_bins
    
  def update(self, pi, V, h, states, actions, rewards, gamma):
    T = len(states)
    dlogits = np.zeros_like(pi)
    dV = np.zeros_like(V)
    dlogits_h = np.zeros_like(h)

    for i in range(T):
      x_s, a_s, r = states[i], actions[i], rewards[i]
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_bin_ind = (np.abs(self._return_bins - G)).argmin()
      hca_factor = (1. - pi[x_s, :] / h[:, x_s, G_bin_ind])
      G_hca = G * hca_factor

      dlogits[x_s, a_s] += G_hca[a_s]
      dlogits[x_s] -= pi[x_s] * G_hca[a_s]
      dV[x_s] += (G - V[x_s])
      dlogits_h[a_s, x_s, G_bin_ind] += 1
      dlogits_h[:, x_s, G_bin_ind] -= h[:, x_s, G_bin_ind]
        
    return dlogits, dV, dlogits_h

import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQN():
    """Agent qui utilise l'algorithme de deep QLearning avec replaybuffer."""

    def __init__(self, dim_etat: int, dim_action: int, gamma: float = 0.99, lr: float = 1e-3, buffer_size: int = 10000, batch_size: int = 64):
        """Constructeur initialisant le réseau, l'optimiseur et le ReplayBuffer"""
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialiser le réseau Q
        self.q_network = QNN(dim_etat, dim_action)
        
        # Initialiser l'optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialiser le replay buffer R 
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def phase_interaction(self, etat: np.ndarray, action: int, recompense: float, etat_suivant: np.ndarray, terminaison: bool):
        """Mémorisation dans le buffer"""
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        
    def action_egreedy(self, etat: np.ndarray, eps: float = 0.0) -> int:
        """Sélectionne une action selon la stratégie epsilon-greedy."""
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(etat)
                return torch.argmax(q_values).item()
                
    def phase_apprentissage(self):
        """Phase d'apprentissage : calcul tensoriel sur un minibatch."""
        if len(self.memory) < self.batch_size:
            return
        
        # Récupération du lot depuis le buffer
        etats, actions, recompenses, etats_suivants, terminaisons = self.memory.sample()
        
        actions = actions.view(-1)
        recompenses = recompenses.view(-1)
        terminaisons = terminaisons.view(-1)
        
        q_pred = self.q_network(etats).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Formule de Bellman
        with torch.no_grad():
            q_suivant_max = self.q_network(etats_suivants).max(1)[0]
            q_cible = recompenses + (self.gamma * q_suivant_max * (1 - terminaisons))
            
        # Calcul de la MSE Loss et descente de gradient
        loss = F.mse_loss(q_pred, q_cible)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
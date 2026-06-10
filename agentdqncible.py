import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from QNN import QNN
from replaybuffer import ReplayBuffer

class AgentDQNCible():
    """Agent qui utilise l'algorithme DQN avec réseau cible."""

    def __init__(self, dim_etat: int, dim_action: int, gamma: float = 0.99, lr: float = 1e-3, 
                 buffer_size: int = 10000, batch_size: int = 64, target_update_freq: int = 300):
        """Constructeur."""
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.learn_step_counter = 0
        
        # Initialiser le réseau Q original
        self.q_network = QNN(dim_etat, dim_action)
        
        # Initialiser le réseau cible
        self.target_network = QNN(dim_etat, dim_action)
        
        # Synchronisation initiale des poids
        self.update_target_network()
        
        # Optimiseur uniquement sur le réseau principal
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialiser le replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def update_target_network(self):
        """Recopie entièrement les poids du réseau original vers le réseau cible."""
        for param_duplicat, param_source in zip(self.target_network.parameters(), self.q_network.parameters()):
            param_duplicat.data.copy_(param_source.data)

    def phase_interaction(self, etat: np.ndarray, action: int, recompense: float, etat_suivant: np.ndarray, terminaison: bool):
        """Mémorisation dans le buffer."""
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
        """Phase d'apprentissage avec calculs tensoriels et cible stable via target_network."""
        if len(self.memory) < self.batch_size:
            return
        
        # Récupération d'un minibatch
        etats, actions, recompenses, etats_suivants, terminaisons = self.memory.sample()
        
        # Ajustement des dimensions
        actions = actions.view(-1)
        recompenses = recompenses.view(-1)
        terminaisons = terminaisons.view(-1)
        
        # Calcul de la prédiction avec le réseau principal
        q_pred = self.q_network(etats).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcul de la cible avec le réseau cible
        with torch.no_grad():
            q_suivant_max = self.target_network(etats_suivants).max(1)[0]
            q_cible = recompenses + (self.gamma * q_suivant_max * (1 - terminaisons))
            
        # Calcul de la perte et descente de gradient
        loss = F.mse_loss(q_pred, q_cible)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mise à jour du compteur et vérification de la synchronisation du réseau cible
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()
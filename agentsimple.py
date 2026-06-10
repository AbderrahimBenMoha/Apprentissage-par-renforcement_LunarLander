import numpy as np
import random
import torch

from QNN import QNN

class AgentSimple():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d'exploration (pas d'apprentissage)."""

    def __init__(self, dim_entree: int, dim_sortie: int):
        self.q_network = QNN(dim_entree, dim_sortie)
        self.dim_sortie = dim_sortie

    def action_egreedy(self, etat : np.ndarray , eps: float = 0.0) -> int:
        if random.random() < eps:
            return random.randint(0, self.dim_sortie - 1)
        
        else:
            with torch.no_grad():
                q_values = self.q_network(etat)
                action = torch.argmax(q_values).item()
            return action
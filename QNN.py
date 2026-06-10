import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self,dim_entree:int, dim_sortie:int):
        """Initialisation des parametres ..."""
        super(QNN, self).__init__()
        
        self.fc1 = nn.Linear(dim_entree, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_sortie)
        
    def forward(self, etat: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float)
            
        if etat.dim() == 1:
            etat = etat.unsqueeze(0)
            
        x = F.relu(self.fc1(etat))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values
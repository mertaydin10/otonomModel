import torch
import torch.nn as nn
import numpy as np

class PPOActorPolicy(nn.Module):
    def __init__(self, state_size=102, hidden_sizes=[256, 256, 128], action_size=5):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh()
        )
        self.action_net = nn.Linear(hidden_sizes[2], action_size)

    def forward(self, x):
        features = self.policy_net(x)
        logits = self.action_net(features)
        return logits

class PPOAgent:
    def __init__(self, state_size=102, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = PPOActorPolicy(state_size=state_size, action_size=action_size)

    def load(self, model_path):
        """
        Loads the PPO actor policy from either policy.pth directly 
        or from a stable-baselines3 directory containing policy.pth.
        """
        import os
        if os.path.isdir(model_path):
            pth_path = os.path.join(model_path, "policy.pth")
        else:
            pth_path = model_path
            
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"PPO policy weight file not found at: {pth_path}")
            
        state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
        
        # Extract actor policy weights
        policy_state_dict = {
            'policy_net.0.weight': state_dict['mlp_extractor.policy_net.0.weight'],
            'policy_net.0.bias': state_dict['mlp_extractor.policy_net.0.bias'],
            'policy_net.2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
            'policy_net.2.bias': state_dict['mlp_extractor.policy_net.2.bias'],
            'policy_net.4.weight': state_dict['mlp_extractor.policy_net.4.weight'],
            'policy_net.4.bias': state_dict['mlp_extractor.policy_net.4.bias'],
            'action_net.weight': state_dict['action_net.weight'],
            'action_net.bias': state_dict['action_net.bias'],
        }
        
        self.policy.load_state_dict(policy_state_dict)
        self.policy.eval()
        print(f"[PPO] Model loaded successfully from {pth_path}")

    def act(self, state):
        """
        Selects the best action (argmax logits) deterministically.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_t)
            action = torch.argmax(logits, dim=1).item()
        return action

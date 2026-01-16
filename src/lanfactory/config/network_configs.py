"""This Module defines simple examples for network and training configurations that serve
as inputs to the training classes in the package.

The configs are organized as follows:
- network_config_mlp / train_config_mlp: For LAN (likelihood approximation networks)
- network_config_choice_prob / train_config_choice_prob: For CPN/OPN (choice probability networks)

For backward compatibility, network_config_cpn, network_config_opn, train_config_cpn,
and train_config_opn are provided as aliases to the choice_prob configs.
"""

# --- Network Configurations ---

# LAN (Likelihood Approximation Network) config
# Output type: logprob (log-probabilities)
network_config_mlp = {
    "layer_sizes": [100, 100, 1],
    "activations": ["tanh", "tanh", "linear"],
    "train_output_type": "logprob",
}

# Choice Probability Network config (used for both CPN and OPN)
# Output type: logits (transformed to log-probabilities during inference)
network_config_choice_prob = {
    "layer_sizes": [100, 100, 1],
    "activations": ["tanh", "tanh", "linear"],
    "train_output_type": "logits",
}

# Backward-compatible aliases
network_config_cpn = network_config_choice_prob
network_config_opn = network_config_choice_prob


# --- Training Configurations ---

# LAN training config
# Loss: huber (for log-probability regression)
train_config_mlp = {
    "cpu_batch_size": 256,
    "gpu_batch_size": 512,
    "n_epochs": 5,
    "optimizer": "adam",
    "learning_rate": 0.002,
    "lr_scheduler": "reduce_on_plateau",
    "lr_scheduler_params": {},
    "weight_decay": 0.0,
    "loss": "huber",
    "save_history": True,
}

# Choice Probability Network training config (used for both CPN and OPN)
# Loss: bcelogit (binary cross-entropy with logits)
train_config_choice_prob = {
    "cpu_batch_size": 256,
    "gpu_batch_size": 512,
    "n_epochs": 5,
    "optimizer": "adam",
    "learning_rate": 0.002,
    "lr_scheduler": "reduce_on_plateau",
    "lr_scheduler_params": {},
    "weight_decay": 0.0,
    "loss": "bcelogit",
    "save_history": True,
}

# Backward-compatible aliases
train_config_cpn = train_config_choice_prob
train_config_opn = train_config_choice_prob

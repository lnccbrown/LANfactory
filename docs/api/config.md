:::lanfactory.config

:::lanfactory.config.network_configs

## Public configuration dictionaries

| Export | Purpose |
| --- | --- |
| `network_config_mlp` | Default LAN MLP architecture |
| `network_config_choice_prob` | Shared choice-probability architecture |
| `network_config_cpn` | Backward-compatible alias of `network_config_choice_prob` |
| `network_config_opn` | Backward-compatible alias of `network_config_choice_prob` |
| `train_config_mlp` | Default LAN training settings |
| `train_config_choice_prob` | Shared choice-probability training settings |
| `train_config_cpn` | Backward-compatible alias of `train_config_choice_prob` |
| `train_config_opn` | Backward-compatible alias of `train_config_choice_prob` |

Copy a dictionary before changing it; the objects exported by the module are
shared mutable defaults.

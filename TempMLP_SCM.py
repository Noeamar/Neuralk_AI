import torch
import torch.nn as nn
import math, random
import numpy as np
from prior.utils import GaussianNoise, XSampler


class TemporalMLPSCM(nn.Module):
    """
    Version temporelle avancée du MLPSCM :
    - Dépendance autoregressive h_t = h_new + alpha * mem[k]
    - Dépendance périodique h_t += beta * mem[k]_{t-period}
    - Injection de bruit gaussien
    - Diversité obtenue via:
        * poids initialisés aléatoirement
        * sampling X/y fixe
        * causes différentes
        * bruit + AR + périodicité
    """

    def __init__(
        self,
        seq_len=500,
        num_features=10,
        num_outputs=1,
        num_causes=10,
        num_layers=4,
        hidden_dim=32,
        mlp_activations=nn.Tanh,
        init_std=1.0,
        block_wise_dropout=True,
        mlp_dropout_prob=0.1,
        scale_init_std_by_dropout=True,
        noise_std=0.01,
        pre_sample_noise_std=False,
        alpha=0.8,
        beta=0.6,
        period=12,
        use_periodicity=True,
        device="cpu",
    ):
        super().__init__()

        # === paramètres identiques au MLPSCM ===
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_causes = num_causes
        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.alpha = alpha
        self.beta = beta
        self.period = period
        self.use_periodicity = use_periodicity
        self.device = device

        # sampler identique au MLPSCM
        self.xsampler = XSampler(seq_len, num_causes, device=device)

        # === architecture temporelle identique au MLPSCM mais adaptée ===
        self.layers = nn.ModuleList()

        # couche 0
        self.layers.append(
            self.generate_layer_modules(
                input_dim=num_causes + hidden_dim,
                output_dim=hidden_dim,
            )
        )

        # couches suivantes
        for _ in range(num_layers - 1):
            self.layers.append(
                self.generate_layer_modules(
                    input_dim=hidden_dim + hidden_dim,
                    output_dim=hidden_dim,
                )
            )

        self.total_hidden_dim = num_layers * hidden_dim

        # indices fixes pour sampling X/y
        self.idx_X, self.idx_y = self.sample_indices_once()

        # initialisation (identique au MLPSCM)
        self.initialize_parameters()

    # ------------------------------------------------------------------
    # === identique au MLPSCM ===
    # ------------------------------------------------------------------
    def generate_layer_modules(self, input_dim, output_dim):
        activation = self.mlp_activations()
        linear_layer = nn.Linear(input_dim, output_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, output_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std

        noise_layer = GaussianNoise(noise_std)
        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self.initialize_with_block_dropout(param, i)
            else:
                self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
            nn.init.normal_(param[block_slice], std=self.init_std / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1))

    def initialize_normally(self, param, index):
        if param.dim() == 2:
            dropout_prob = self.mlp_dropout_prob if index > 0 else 0
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / ((1 - dropout_prob)**0.5 if self.scale_init_std_by_dropout else 1)
            nn.init.normal_(param, std=std)
            param.data = param.data * torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    def sample_indices_once(self):
        perm = torch.randperm(self.total_hidden_dim, device=self.device)
        return perm[:self.num_features], perm[-self.num_outputs:]

    # ------------------------------------------------------------------
    # === forward temporel + périodicité + bruit ===
    # ------------------------------------------------------------------
    def forward(self):
        T = self.seq_len
        causes = self.xsampler.sample()

        mem = [torch.zeros(self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
        past_states = [torch.zeros((T, self.hidden_dim), device=self.device) for _ in range(self.num_layers)]

        X = torch.zeros(T, self.num_features)
        y = torch.zeros(T, self.num_outputs)

        for t in range(T):
            h_prev = None
            layer_states = []

            for k in range(self.num_layers):
                if k == 0:
                    inp = torch.cat([causes[t], mem[k]])
                else:
                    inp = torch.cat([h_prev, mem[k]])

                h_new = self.layers[k](inp)
                periodic_term = 0

                if self.use_periodicity and t >= self.period:
                    periodic_term = self.beta * past_states[k][t - self.period]

                h_t = (
                    h_new
                    + self.alpha * mem[k]
                    + periodic_term
                    + 0.1 * torch.randn_like(h_new)
                )

                if torch.any(torch.isnan(h_t)) or torch.any(torch.isinf(h_t)):
                    h_t = torch.zeros_like(h_t)

                # clip pour éviter l’explosion
                h_t = torch.clamp(h_t, -50, 50)

                mem[k] = h_t
                past_states[k][t] = h_t
                h_prev = h_t
                layer_states.append(h_t)

            out = torch.cat(layer_states)
            X[t] = out[self.idx_X]
            y[t] = out[self.idx_y]

        if self.num_outputs == 1:
            y = y.squeeze(-1)
        return X, y

    # ------------------------------------------------------------------
    # === Fonction batch pour construire un dataset complet ===
    # ------------------------------------------------------------------
    def generate_dataset(self, n_individuals=50):
        """
        Retourne un dataset où chaque individu est empilé verticalement.
        Final shape :
            X : (n_individuals * seq_len, num_features)
            y : (n_individuals * seq_len, num_outputs)
        """
        X_list = []
        y_list = []

        for _ in range(n_individuals):
            X_i, y_i = self.forward()
            X_list.append(X_i)
            y_list.append(y_i.unsqueeze(1))  # pour unifier dimensions

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        return X, y
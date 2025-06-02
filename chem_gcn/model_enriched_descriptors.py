"""
Script modified from: 
    G. S. Deshmukh, “ChemGCN: A Graph Convolutional Network for Chemical Property
    Prediction,” GitHub repository, 2020. Available: https://github.com/gauravsdeshmukh/ChemGCN
"""
import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    def __init__(self, node_in_len: int, node_out_len: int):
        super().__init__()
        self.conv_linear = nn.Linear(node_in_len, node_out_len)
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True) + 1e-6 
        idx_mat = torch.eye(adj_mat.shape[-2], adj_mat.shape[-1], device=n_neighbors.device)
        idx_mat = idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)
        node_fea = self.conv_linear(node_fea)
        node_fea = self.conv_activation(node_fea)

        return node_fea

class PoolingLayer(nn.Module):
    def forward(self, node_fea):
        return node_fea.mean(dim=1)

class ChemGCN(nn.Module):
    def __init__(self, node_vec_len, node_fea_len, hidden_fea_len, 
                 n_conv, n_hidden, molecular_descriptors, n_outputs, p_dropout=0.0):
        super().__init__()

        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        self.conv_layers = nn.ModuleList([
            ConvolutionLayer(node_fea_len, node_fea_len) for _ in range(n_conv)
        ])

        self.pooling = PoolingLayer()
        self.pooling_activation = nn.LeakyReLU()

        self.pooled_to_hidden = nn.Linear(node_fea_len + molecular_descriptors, hidden_fea_len)
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_fea_len)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_fea_len, hidden_fea_len) for _ in range(n_hidden - 1)
        ])

        self.hidden_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)

        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat, molecular_descriptors):
        node_fea = self.init_transform(node_mat)

        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)

        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)

        combined_features = torch.cat((pooled_node_fea, molecular_descriptors), dim=-1)
        neural_fingerprints = combined_features.clone().detach()

        hidden_node_fea = self.pooled_to_hidden(combined_features)
        hidden_node_fea = self.batch_norm_hidden(hidden_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)

        for i in range(len(self.hidden_layers)):
            residual = hidden_node_fea  # Add residual connection
            hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
            hidden_node_fea = self.hidden_activation(hidden_node_fea)
            if i % 2 == 0:  
                hidden_node_fea = self.dropout(hidden_node_fea)
            hidden_node_fea += residual  # Residual connection

        out = self.hidden_to_output(hidden_node_fea)
        return out, neural_fingerprints

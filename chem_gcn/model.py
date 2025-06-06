"""
Script modified from: 
    G. S. Deshmukh, “ChemGCN: A Graph Convolutional Network for Chemical Property
    Prediction,” GitHub repository, 2020. Available: https://github.com/gauravsdeshmukh/ChemGCN
"""
import torch
import torch.nn as nn

#### CLASSES
class ConvolutionLayer(nn.Module):
    def __init__(self, node_in_len: int, node_out_len: int):
        #Call constructor of base class
        super().__init__()

        #Create linear layer for node matrix
        self.conv_linear = nn.Linear(node_in_len, node_out_len)

        #Create activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        #Calculate number of neighbors
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)

        #Create identity tensor
        self.idx_mat = torch.eye(adj_mat.shape[-2], adj_mat.shape[-1], device=n_neighbors.device)

        #Add new batch dimension and expand
        idx_mat = self.idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        #Get inverse degree matrix
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        #Perform matrix multiplication: D**(-1)AN
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)

        #Perform linear transformation to node features (multiplication with W)
        node_fea = self.conv_linear(node_fea)

        #Apply activation
        node_fea = self.conv_activation(node_fea)

        return node_fea
    
class PoolingLayer(nn.Module):
    def __init__(self):
        #Call constructor of base class
        super().__init__()

    def forward(self, node_fea):
        #Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea
    
class ChemGCN(nn.Module): #With Batch Norm
    def __init__(self, node_vec_len: int, 
                 node_fea_len: int, hidden_fea_len: int, 
                 n_conv: int, n_hidden: int, n_outputs: int, p_dropout: float = 0.0,
                 ):
        #Call constructor of base class
        super().__init__()

        #Define layers
        #Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        #Convolution layers
        self.conv_layers = nn.ModuleList(
            [
                ConvolutionLayer(
                    node_in_len = node_fea_len,
                    node_out_len = node_fea_len,
                )
                for i in range(n_conv)
            ]
        )

        #Pool convolutional outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        #Pooling activation
        self.pooling_activation = nn.LeakyReLU()

        #From pooled vector to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        ###ADDED###
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_fea_len)

        #Hidden layer
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)

        #Hidden layer activation function
        self.hidden_activation = nn.LeakyReLU()

        #Hidden later dropout
        self.dropout = nn.Dropout(p=p_dropout)

        #If hidden layers more than 1, add more hidden layers
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_fea_len, hidden_fea_len) for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
            )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        #Final later going to the output
        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)


    def forward(self, node_mat, adj_mat):
        #Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)

        #Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)
        
        #Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)

        # Save neural fingerprints
        neural_fingerprints = pooled_node_fea.clone().detach()

        #First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        ###ADDED#
        hidden_node_fea = self.batch_norm_hidden(hidden_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        #Subsequent hidden layers
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)
        
        #Output
        out = self.hidden_to_output(hidden_node_fea)

        return out, neural_fingerprints

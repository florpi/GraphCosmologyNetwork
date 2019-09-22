import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from fully_connected import FullyConnectedModel


def gcn_message(edges, weights: bool = False):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    if weights:
        return {"msg": edges.src["h"] * edges.data["weights"]}
    else:
        return {"msg": edges.src["h"]}


def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {"h": torch.sum(nodes.mailbox["msg"], dim=1)}


# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, weights=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.weights = weights

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata["h"] = inputs
        # trigger message passing on all edges
        weighted_gcn_message = lambda edges: gcn_message(edges, self.weights)
        g.send(g.edges(), weighted_gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop("h")
        # perform linear transformation
        return self.linear(h)


# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(
        self,
        input_features: torch.Tensor,
        weights: bool = False,
        conv_hidden_dim: int = 512,
        embedding_dim: int = 512,
        readout_n_hidden_layers: int = 2,
        readout_hidden_dim: int = 512,
        n_output_dim: int = 2,
    ):

        super(GCN, self).__init__()

        self.gcn1 = GCNLayer(input_features, conv_hidden_dim, weights)
        self.gcn2 = GCNLayer(conv_hidden_dim, embedding_dim, weights)

        self.fcn = FullyConnectedModel(
            n_features=embedding_dim,
            n_hidden_layers=readout_n_hidden_layers,
            n_hidden_dim=readout_hidden_dim,
            n_output_dim=n_output_dim,
        )

    def forward(self, g, inputs):

        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        h = torch.relu(h)

        # concatenate embedding and features of the node
        # h = torch.cat([h, inputs], dim = -1)
        # h += inputs

        # Readout layers:
        h = self.fcn(h)
        return h


# -----------------------------------------------------------------------------
# TEST MODEL
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    maximum_distance = 10  # Mpc
    learning_rate = 1.0e-3
    num_epochs = 1
    conv_hidden_dim = 512
    embedding_dim = 512
    readout_hidden_size = 512
    n_classes = 2
    n_nodes = 3

    G = dgl.DGLGraph()

    G.add_nodes(n_nodes)
    G.ndata["h"] = torch.zeros((n_nodes, 5))
    G.ndata["h"][0, 0] = 1.0
    G.ndata["h"][1, 1] = 2.0
    G.ndata["h"][2, 2] = 3.0

    print(f"Input Graph with {n_nodes} nodes")
    print(G.ndata["h"])

    edges_in = [0, 1, 2]  # Need to add self loop to include its own features!
    edges_out = 2
    G.add_edges(edges_in, edges_out)  # 0->2, 1->2

    print("Added following edges:")

    print(f"{edges_in[0]} -> {edges_out}")
    print(f"{edges_in[1]} -> {edges_out}")
    print(f"{edges_in[2]} -> {edges_out}")

    print("Message aggregated at the different nodes")

    """
	conv_hidden_dim = G.ndata["h"].shape[-1]

	G.send(G.edges(), gcn_message)

	G.recv(G.nodes(), gcn_reduce)
	aggregated_message = G.ndata['h']

	print(f'Aggregated message has shape : {aggregated_message.shape}')
	"""

    # Compute the forward pass through the model
    print("Computing forward pass...", end=" ", flush=True)

    net = GCN(
        G.ndata["h"].shape[-1],
        False,
        conv_hidden_dim,
        embedding_dim,
        readout_hidden_size,
        n_output_dim=2,
    )

    output = net(G, G.ndata["h"])

    print("Done!", flush=True)
    print("Output shape:", output.shape)

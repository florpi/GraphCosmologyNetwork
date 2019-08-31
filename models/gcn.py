import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the message & reduce function
# NOTE: we ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
	# The argument is a batch of edges.
	# This computes a (batch of) message called 'msg' using the source node's feature 'h'.
	return {"msg": edges.src["h"] }#* edges.data["inv_dist"]}


def gcn_reduce(nodes):
	# The argument is a batch of nodes.
	# This computes the new 'h' features by summing received 'msg' in each node's mailbox.
	# TODO: Check itself is there !
	return {"h": torch.sum(nodes.mailbox["msg"], dim=1)}


# Define the GCNLayer module
class GCNLayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GCNLayer, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)

	def forward(self, g, inputs):
		# g is the graph and the inputs is the input node features
		# first set the node features
		g.ndata["h"] = inputs
		# trigger message passing on all edges
		g.send(g.edges(), gcn_message)
		# trigger aggregation at all nodes
		g.recv(g.nodes(), gcn_reduce)
		# get the result node features
		h = g.ndata.pop("h")
		# perform linear transformation
		return self.linear(h)


# Define a 2-layer GCN model
class GCN(nn.Module):
	def __init__(self, in_feats, conv_hidden_size, embedding_size, readout_hidden_size, num_classes = 1):
		super(GCN, self).__init__()
		self.gcn1 = GCNLayer(in_feats, conv_hidden_size)
		#self.gcn2 = GCNLayer(conv_hidden_size, embedding_size)

		self.fcn1 = nn.Linear(embedding_size, readout_hidden_size)
		self.fcn2 = nn.Linear(readout_hidden_size, num_classes)

	def forward(self, g, inputs):

		h = self.gcn1(g, inputs)
		h = torch.relu(h)
		#h = self.gcn2(g, h)
		#h = torch.relu(h)

		# concatenate embedding and features of the node
		#h = torch.cat([h, inputs], dim = -1)
		#h += inputs

		# Readout layers:
		h = self.fcn1(h)
		h = F.relu(h)
		h = self.fcn2(h)
		return h

if __name__ == '__main__':

	maximum_distance = 10  # Mpc
	learning_rate = 1.0e-3
	num_epochs = 1
	readout_hidden_size = 512
	n_classes = 2
	n_nodes = 3



	G = dgl.DGLGraph()

	G.add_nodes(n_nodes)
	G.ndata['h'] = torch.zeros((n_nodes, 5))  
	G.ndata['h'][0, 0] = 1.
	G.ndata['h'][1, 1] = 2.
	G.ndata['h'][2, 2] = 3.

	print(f'Input Graph with {n_nodes} nodes')
	print(G.ndata['h'])
	
	edges_in = [0, 1, 2] # Need to add self loop to include its own features!
	edges_out = 2
	G.add_edges(edges_in, edges_out)  # 0->2, 1->2

	print('Added following edges:')

	print(f'{edges_in[0]} -> {edges_out}')
	print(f'{edges_in[1]} -> {edges_out}')
	print(f'{edges_in[2]} -> {edges_out}')

	print('Message aggregated at the different nodes')

	conv_hidden_size = G.ndata["h"].shape[-1] 

	G.send(G.edges(), gcn_message)

	G.recv(G.nodes(), gcn_reduce)
	print(G.ndata['h'])




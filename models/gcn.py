import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the message & reduce function
# NOTE: we ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
	# The argument is a batch of edges.
	# This computes a (batch of) message called 'msg' using the source node's feature 'h'.
	print(edges.src["h"].shape)
	print(edges.src["h"])
	return {"msg": edges.src["h"] }#* edges.data["inv_dist"]}


def gcn_reduce(nodes):
	# The argument is a batch of nodes.
	# This computes the new 'h' features by summing received 'msg' in each node's mailbox.
	# TODO: Check itself is there !
	print('mailbox')
	print(nodes.mailbox["msg"].shape)
	print(nodes.mailbox["msg"])
	print(torch.sum(nodes.mailbox["msg"], dim = -1).shape)
	print(torch.sum(nodes.mailbox["msg"], dim = -1))
	return {"h": torch.sum(nodes.mailbox["msg"], dim=-1)}


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

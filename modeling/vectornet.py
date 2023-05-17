import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes
from utils.func_lab import get_from_mapping, merge_tensors, de_merge_tensors


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = 3

        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = merge_tensors(input_list, device, self.hidden_size)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(de_merge_tensors(hidden_states, lengths))


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, cfg, device):
        super(VectorNet, self).__init__()
        self.cfg = cfg
        self.device = device
        hidden_size = cfg.MODEL.hidden_size
        self.hidden_size = hidden_size

        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)

        # Use multi-head attention and residual connection.
        self.global_graph = GlobalGraphRes(hidden_size)
        self.laneGCN_A2L = CrossAttention(hidden_size)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size)
        self.laneGCN_L2A = CrossAttention(hidden_size)

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, device, batch_size):
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        # matrix=list[bs]=np[n,128]
        # polyline_spans=list[bs]=list[n_ped]=slice

        input_list_list = []
        map_input_list_list = []
        lane_input_mask_list = []  # list[bs]
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []  # ped vector
            map_input_list = []  # lane vector
            lane_input_mask = []  # list[n_lane]
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']  # n_ped
            for j, polyline_span in enumerate(polyline_spans[i]):
                vector = matrix[i][polyline_span]  # [n,128]
                mask = np.sum(np.isnan(vector), axis=-1) == 0  # [n]有效的true
                if np.sum(mask) == 0:
                    tensor = torch.zeros((1, self.hidden_size), dtype=torch.float, device=device)
                    if j >= map_start_polyline_idx:
                        map_input_list.append(tensor)
                        lane_input_mask.append(True)  # true表示需要mask掉的
                else:
                    tensor = torch.tensor(vector[mask, :], dtype=torch.float, device=device)  # [n',128], n'表示一个ped的t
                    input_list.append(tensor)
                    if j >= map_start_polyline_idx:
                        map_input_list.append(tensor)
                        lane_input_mask.append(False)

            input_list_list.append(input_list)  # list[bs]=list[n_ped+n_lane]=tensor[n',128]
            map_input_list_list.append(map_input_list)  # list[bs]=list[n_lane]=tensor[n',128] 好像一直都是[9,128]
            lane_input_mask_list.append(lane_input_mask)  # list[bs]=list[n_lane]=bool

        element_states_batch = []  # ped and lane
        for i in range(batch_size):
            a, b = self.point_level_sub_graph(input_list_list[i])
            element_states_batch.append(a)  # a=[n_ped+n_lane,128]

        lane_states_batch = []  # lane
        for i in range(batch_size):
            a, b = self.point_level_sub_graph(map_input_list_list[i])
            lane_states_batch.append(a)  # a=[n_lane,128]

        # We follow laneGCN to fuse realtime traffic information from agent nodes to lane nodes.
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            agents = element_states_batch[i][:map_start_polyline_idx]  # [?,128]
            lanes = element_states_batch[i][map_start_polyline_idx:]  # [?,128]
            # Origin laneGCN contains three fusion layers. Here one fusion layer is enough.
            lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)  # 用agents[0:1]也就是目标人，与lane做cross attention
            element_states_batch[i] = torch.cat([agents, lanes])

        return element_states_batch, lane_states_batch, lane_input_mask_list  # list[bs]=[n_ped+n_lane,128], list[bs]=[n_lane,128], list[bs]=[n]

    def forward(self, mapping, device, infer=False):
        matrix = get_from_mapping(mapping, 'matrix')  # list[bs]=np[n,128]
        polyline_spans = get_from_mapping(mapping, 'polyline_spans')  # start-end信息, list[bs]=list[n_ped]=slice

        batch_size = len(matrix)

        element_states_batch, lane_states_batch, lane_input_mask_list = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)  # list[bs]=[n_ped+n_lane,128], list[bs]=[n_lane,128], list[bs]=[n_lane]

        inputs, inputs_lengths = merge_tensors(element_states_batch, device, self.hidden_size)  # [bs,max_p_n,128], list[bs]=int
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)  # [bs,max_p_n,128]  实际上只有[:,0,:]上了

        return lane_states_batch, lane_input_mask_list, inputs, inputs_lengths, hidden_states
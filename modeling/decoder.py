import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

from modeling.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP

from utils.func_lab import get_neighbour_points, get_dis, get_from_mapping


class DecoderRes(nn.Module):  # 全连接+残差
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):  # 全连接+残差
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class TrajCompleter(nn.Module):
    def __init__(self, device, hidden_size, future_frame_num):
        super(TrajCompleter, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.future_frame_num = future_frame_num

        self.goals_2D_mlps = nn.Sequential(
            MLP(2, hidden_size),
            MLP(hidden_size),
            MLP(hidden_size)
        )

        self.complete_traj_cross_attention = CrossAttention(hidden_size)
        self.complete_traj_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.future_frame_num * 2)

    def forward(self, target, f_input, hidden_state, infer):
        # target=[k,2], f_input=[?,128], hidden_state=[?,128]
        k = target.shape[0]
        target_feature = self.goals_2D_mlps(torch.tensor(target, dtype=torch.float, device=self.device))  # [k,128]  这个模块不参与训练，仅仅作为一个固定参数的mapping，然后在eval的时候把预测的goal生成轨迹
        # if not infer:
        target_feature.detach_()
        hidden_attention = self.complete_traj_cross_attention(
            target_feature.unsqueeze(0), f_input.detach().unsqueeze(0)
        ).squeeze(0)  # [k,128], [n_lane+n_pen,128] -> [k,128]
        predict_traj = self.complete_traj_decoder(
            torch.cat([hidden_state[0:1, :].detach().expand(k, -1), target_feature, hidden_attention], dim=-1)
        ).view([k, self.future_frame_num, 2])  # [k,dim] -> [k,30,2]

        return predict_traj  # [k,30,2]

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.MODEL.hidden_size
        self.future_frame_num = cfg.DATA.pred_len
        self.mode_num = cfg.MODEL.mode_num

        self.decoder = DecoderRes(hidden_size, out_features=2)

        # if 'variety_loss' in args.other_params:
        #     self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2)
        #
        #     if 'variety_loss-prob' in args.other_params:
        #         self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2 + 6)

        # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
        # self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
        self.goals_2D_cross_attention = CrossAttention(hidden_size)
        # Fuse goal feature and agent feature when encoding goals.
        self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)

        self.stage_one_cross_attention = CrossAttention(hidden_size)
        self.stage_one_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
        self.stage_one_goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)

    def lane_scoring(self, i, mapping, lane_states_batch, lane_input_mask, inputs, inputs_lengths,
                     hidden_states, device, loss):
        def get_stage_one_scores():
            stage_one_hidden = lane_states_batch[i]
            stage_one_hidden_attention = self.stage_one_cross_attention(
                stage_one_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)  # [n_lane,128], [n_lane+n_ped,128] -> [n_lane,128]
            stage_one_scores = self.stage_one_decoder(torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(
                stage_one_hidden.shape), stage_one_hidden, stage_one_hidden_attention], dim=-1))  # [n_lane,1]
            stage_one_scores = stage_one_scores.squeeze(-1)
            mask = torch.tensor(lane_input_mask[i], dtype=torch.bool, device=device)
            stage_one_scores.masked_fill(mask, -np.inf)
            stage_one_scores = F.log_softmax(stage_one_scores, dim=-1)
            return stage_one_scores

        stage_one_scores = get_stage_one_scores()  # [n_lane]
        assert len(stage_one_scores) == len(mapping[i]['polygons'])
        loss[i] += F.nll_loss(stage_one_scores.unsqueeze(0),
                              torch.tensor([mapping[i]['stage_one_label']], device=device))

        # Select top K lanes, where K is dynamic. The sum of the probabilities of selected lanes is larger than threshold (0.95).
        _, stage_one_topk_ids = torch.topk(stage_one_scores, k=len(stage_one_scores))
        threshold = 0.95
        sum = 0.0
        for idx, each in enumerate(torch.exp(stage_one_scores[stage_one_topk_ids])):
            sum += each
            if sum > threshold:
                stage_one_topk_ids = stage_one_topk_ids[:idx + 1]
                break
        # todo: add 作用是打印len(stage_one_topk_ids)
        # utils.other_errors_put('stage_one_k', len(stage_one_topk_ids))
        # _, stage_one_topk_ids = torch.topk(stage_one_scores, k=min(args.stage_one_K, len(stage_one_scores)))
        # if mapping[i]['stage_one_label'] in stage_one_topk_ids.tolist():
        #     utils.other_errors_put('stage_one_recall', 1.0)
        # else:
        #     utils.other_errors_put('stage_one_recall', 0.0)

        topk_lanes = lane_states_batch[i][stage_one_topk_ids]
        return topk_lanes

    def get_scores_of_dense_goals(self, i, goals_2D, mapping, labels, device, scores,
                                  get_scores_inputs, gt_points=None):
        k = 150
        _, topk_ids = torch.topk(scores, k=min(k, len(scores)))
        topk_ids = topk_ids.tolist()

        # Sample dense goals from top K sparse goals.
        goals_2D_dense = get_neighbour_points(goals_2D[topk_ids], topk_ids=topk_ids, mapping=mapping[i])

        goals_2D_dense = torch.cat([torch.tensor(goals_2D_dense, device=device, dtype=torch.float),
                                    torch.tensor(goals_2D, device=device, dtype=torch.float)], dim=0)

        old_vector_num = len(goals_2D)

        goals_mask = torch.zeros(goals_2D_dense.shape[0]).to(torch.bool).to(device)

        scores = self.get_scores(goals_2D_dense, goals_mask, *get_scores_inputs)

        index = torch.argmax(scores).item()
        point = np.array(goals_2D_dense[index].tolist())

        label = np.array(labels[i]).reshape([self.future_frame_num, 2])
        final_idx = mapping[i].get('final_idx', -1)
        mapping[i]['goals_2D_labels'] = np.argmin(get_dis(goals_2D_dense.detach().cpu().numpy(), label[final_idx]))

        return scores, point, goals_2D_dense.detach().cpu().numpy()


    def goals_2D_per_example(self, i, goals_2D, mapping, lane_states_batch, lane_input_mask,
                             inputs, inputs_lengths, hidden_states, labels,
                             labels_is_valid, device, loss, infer):
        """
        :param i: example index in batch
        :param goals_2D: candidate goals sampled from map (shape ['goal num', 2])
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param loss: (shape [batch_size])
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """

        # i表示batch中的位置
        # goals_2D=[n_goal,2]表示场景中所有不重复的车道中心点
        # mapping
        # lane_states_batch=list[bs]=[n_lane,128] 车道特征
        # lane_input_mask=list[bs]=[n] 表示lane有效无效的mask
        # inputs=[bs,128,128]，对齐之后的所有特征
        # inputs_lengths=list[bs]=int，对齐长度
        # hidden_states=[bs,max_p_n,128]全局特征
        # labels=list[bs]=[30,2]真实的未来轨迹
        # labels_is_valid=list[bs]=ones[30]

        if not infer:
            final_idx = mapping[i].get('final_idx', -1)  # -1
            assert labels_is_valid[i][final_idx]

        gt_points = labels[i].reshape([self.future_frame_num, 2])  # [30,2]

        topk_lanes = None
        # Get top K lanes with the highest probability.
        topk_lanes = self.lane_scoring(i, mapping, lane_states_batch, lane_input_mask, inputs, inputs_lengths,
                                       hidden_states, device, loss)  # [?,128]

        get_scores_inputs = (inputs, hidden_states, inputs_lengths, i, mapping, device, topk_lanes)

        # There is a lane scoring module (see Section 3.2) in the paper in order to reduce the number of goal candidates.
        # In this implementation, we use goal scoring instead of lane scoring, because we observed that it performs slightly better than lane scoring.
        # Here goals_2D are sparse cnadidate goals sampled from map.
        goals_mask = np.isnan(goals_2D).any(axis=-1)  # [n_goal] true表示需要mask掉
        goals_2D[goals_mask, :] = 0.0  # nan改成用0填充
        goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)  # [n_goal,2]
        goals_mask = torch.tensor(goals_mask, dtype=torch.bool, device=device)
        scores = self.get_scores(goals_2D_tensor, goals_mask, *get_scores_inputs)  # [n_goal]打分，给每个goal打分
        index = torch.argmax(scores).item()  # 找到max分的index
        highest_goal = goals_2D[index]
        # 无效的goals不输入dense，因为dense过程中会根据有效dense goal来求label，所以dense可以不用mask，另外存到mapping中也不用mask了，因为这个mask过了
        goals_2D = goals_2D[~goals_mask.detach().cpu().numpy(), :]
        scores = scores[~goals_mask]  # todo: get_scores是不是可以不用mask，因为dense goal不需要mask，而goal的结果出来又被mask掉

        # Get dense goals and their scores.
        # With the help of the above goal scoring, we can reduce the number of dense goals.
        # After this step, goals_2D become dense goals.
        scores, highest_goal, goals_2D = \
            self.get_scores_of_dense_goals(i, goals_2D, mapping, labels, device, scores,
                                           get_scores_inputs, gt_points)  # 变成了dense的goal, scores=[d_goal], highest_goal=[2], goals_2D=[d_goal,2]

        mapping[i]['vis.goals_2D'] = goals_2D
        mapping[i]['vis.scores'] = np.array(scores.tolist())
        mapping[i]['vis.labels'] = gt_points
        mapping[i]['vis.labels_is_valid'] = labels_is_valid[i]

        if infer:
            mapping[i]['goals_2D_scores'] = (goals_2D.astype(np.float32), np.array(scores.tolist(), dtype=np.float32))

        return highest_goal, scores, goals_2D  # np=[2], [d_goals], [d_goals,2]

    # def variety_loss(self, mapping, hidden_states, batch_size, inputs,
    #                  inputs_lengths, labels_is_valid, loss,
    #                  DE, device, labels):
    #     """
    #     :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
    #     :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
    #     :param inputs_lengths: valid element number of each example
    #     :param DE: displacement error (shape [batch_size, self.future_frame_num])
    #     """
    #     outputs = self.variety_loss_decoder(hidden_states[:, 0, :])
    #     pred_probs = None
    #     if 'variety_loss-prob' in args.other_params:
    #         pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
    #         outputs = outputs[:, :-6].view([batch_size, 6, self.future_frame_num, 2])
    #     else:
    #         outputs = outputs.view([batch_size, 6, self.future_frame_num, 2])
    #
    #     for i in range(batch_size):
    #         if args.do_train:
    #             assert labels_is_valid[i][-1]
    #         gt_points = np.array(labels[i]).reshape([self.future_frame_num, 2])
    #         argmin = np.argmin(utils.get_dis_point_2_points(gt_points[-1], np.array(outputs[i, :, -1, :].tolist())))
    #
    #         loss_ = F.smooth_l1_loss(outputs[i, argmin],
    #                                  torch.tensor(gt_points, device=device, dtype=torch.float), reduction='none')
    #         loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_frame_num, 1)
    #         if labels_is_valid[i].sum() > utils.eps:
    #             loss[i] += loss_.sum() / labels_is_valid[i].sum()
    #
    #         if 'variety_loss-prob' in args.other_params:
    #             loss[i] += F.nll_loss(pred_probs[i].unsqueeze(0), torch.tensor([argmin], device=device))
    #     if args.do_eval:
    #         outputs = np.array(outputs.tolist())
    #         pred_probs = np.array(pred_probs.tolist(), dtype=np.float32) if pred_probs is not None else pred_probs
    #         for i in range(batch_size):
    #             for each in outputs[i]:
    #                 utils.to_origin_coordinate(each, i)
    #
    #         return outputs, pred_probs, None
    #     return loss.mean(), DE, None

    def forward(self, mapping, batch_size, lane_states_batch, lane_input_mask, inputs,
                inputs_lengths, hidden_states, device, infer):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        labels = get_from_mapping(mapping, 'labels')
        labels_is_valid = get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        highest_goal_list = []
        scores_list = []

        # if 'variety_loss' in args.other_params:
        #     return self.variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels)

        for i in range(batch_size):
            goals_2D = mapping[i]['goals_2D']

            highest_goal, scores, goals_2D = self.goals_2D_per_example(
                i, goals_2D, mapping, lane_states_batch, lane_input_mask,inputs, inputs_lengths, hidden_states,
                labels, labels_is_valid, device, loss, infer
            )
            highest_goal_list.append(highest_goal)
            scores_list.append(scores)

        # if False:  # 改成开关（作用是可视化）
        #     for i in range(batch_size):
        #         predict = np.zeros((self.mode_num, self.future_frame_num, 2))
        #         utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'],
        #                                  self.future_frame_num,
        #                                  labels=mapping[i]['vis.labels'],
        #                                  labels_is_valid=mapping[i]['vis.labels_is_valid'],
        #                                  predict=predict)
        return loss, highest_goal_list, scores_list

    def get_scores(self, goals_2D_tensor, goals_mask, inputs, hidden_states, inputs_lengths, i, mapping, device, topk_lanes):
        """
        :param goals_2D_tensor: candidate goals sampled from map (shape ['goal num', 2])
        :return: log scores of goals (shape ['goal num'])
        """
        # Fuse goal feature and agent feature when encoding goals.
        goals_2D_hidden = self.goals_2D_point_sub_graph(goals_2D_tensor.unsqueeze(0), hidden_states[i, 0:1, :]).squeeze(0)  # [n_goal,2] -> [n_goal,128]

        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)  # [n_goal,128], [n_lane+n_pen,128] -> [n_goal,128] ->

        # Perform cross attention from goals to top K lanes. It's a trick to improve modeling performance.
        stage_one_goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), topk_lanes.unsqueeze(0)).squeeze(0)
        li = [hidden_states[i, 0, :].unsqueeze(0).expand(goals_2D_hidden.shape),
              goals_2D_hidden, goals_2D_hidden_attention, stage_one_goals_2D_hidden_attention]

        scores = self.stage_one_goals_2D_decoder(torch.cat(li, dim=-1))

        scores = scores.squeeze(-1)
        scores.masked_fill(goals_mask, -np.inf)
        scores = F.log_softmax(scores, dim=-1)
        return scores  # [n_goals]

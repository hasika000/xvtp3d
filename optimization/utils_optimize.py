import numpy as np
import utils_cython
import multiprocessing
from multiprocessing import Process
from utils.func_lab import get_from_mapping, get_dis_point_2_points, to_origin_coordinate, batch_init_origin


def run_process(cfg, queue, queue_res):
    objective = 'MR'
    if cfg.EVAL.MRminFDE != -1:
        objective = 'MRminFDE'
    opti_time = float(cfg.EVAL.opti_time)

    for value in queue:
        idx_in_batch, file_name, (goals_2D, scores), kwargs = value
        scores = np.exp(scores)

        if cfg.EVAL.MRminFDE != -1:
            assert cfg.EVAL.cnt_sample != -1
            MRratio = float(cfg.EVAL.MRminFDE)  # MRminFDE取1.0，MRminFDE=0.0取0.0

        if cfg.EVAL.cnt_sample != -1:
            num_step = 1000
            kwargs.update(dict(
                num_step=num_step,
                cnt_sample=cfg.EVAL.cnt_sample,
                MRratio=MRratio,
            ))
            assert cfg.EVAL.cnt_sample > 1

        results = utils_cython.get_optimal_targets(goals_2D, scores, file_name, objective, opti_time, kwargs=kwargs)

        expectation, ans_points, pred_probs = results
        queue_res.append((idx_in_batch, expectation, ans_points, pred_probs))


def run_process_dist(cfg, queue, queue_res):
    objective = 'MR'
    if cfg.EVAL.MRminFDE != -1:
        objective = 'MRminFDE'
    opti_time = float(cfg.EVAL.opti_time)

    while True:
        value = queue.get()
        if value is None:
            break
        idx_in_batch, file_name, (goals_2D, scores), kwargs = value
        scores = np.exp(scores)

        if cfg.EVAL.MRminFDE != -1:
            assert cfg.EVAL.cnt_sample != -1
            MRratio = float(cfg.EVAL.MRminFDE)  # MRminFDE取1.0，MRminFDE=0.0取0.0

        if cfg.EVAL.cnt_sample != -1:
            num_step = 1000
            kwargs.update(dict(
                num_step=num_step,
                cnt_sample=cfg.EVAL.cnt_sample,
                MRratio=MRratio,
            ))
            assert cfg.EVAL.cnt_sample > 1

        results = utils_cython.get_optimal_targets(goals_2D, scores, file_name, objective, opti_time, kwargs=kwargs)

        expectation, ans_points, pred_probs = results
        queue_res.put((idx_in_batch, expectation, ans_points, pred_probs))
    pass


def select_goals_by_optimization(cfg, batch_gt_points, mapping, goals_scores_tuple, close=False):
    method2FDE = []
    if cfg.EVAL.core_num == 1:

        batch_size, future_frame_num, _ = batch_gt_points.shape

        batch_file_name = get_from_mapping(mapping, 'file_name')

        # init queue
        queue = []
        queue_res = []
        run_times = 8  # batch中每一个，test 8次，从中找最好的
        for _ in range(run_times):
            for i in range(batch_size):
                kwargs = {}

                queue.append((i, batch_file_name[i], goals_scores_tuple[i], kwargs))

        run_process(cfg, queue, queue_res)

        expectations = np.ones(batch_size) * 10000.0
        batch_ans_points = np.zeros([batch_size, 6, 3])
        batch_pred_probs = np.zeros([batch_size, 6])
        assert int(run_times * batch_size) == len(queue_res)
        for idx in range(run_times * batch_size):  # 寻找batch中每个，在8个run_time中最好的值
            i, expectation, ans_points, pred_probs = queue_res[idx]
            if expectation < expectations[i]:
                expectations[i] = expectation
                batch_ans_points[i, :, 0:2] = ans_points
                batch_pred_probs[i] = pred_probs

    else:
        this = select_goals_by_optimization
        if not hasattr(this, 'processes'):
            queue = multiprocessing.Queue(cfg.EVAL.core_num)
            queue_res = multiprocessing.Queue()
            processes = [
                Process(target=run_process_dist, args=(cfg, queue, queue_res))
                for _ in range(cfg.EVAL.core_num)]
            for each in processes:
                each.start()
            this.processes = processes
            this.queue = queue
            this.queue_res = queue_res

        queue = this.queue
        queue_res = this.queue_res

        if close:
            for i in range(cfg.EVAL.core_num):
                queue.put(None)
            for each in select_goals_by_optimization.processes:
                each.join()
            return

        batch_size, future_frame_num, _ = batch_gt_points.shape

        batch_file_name = get_from_mapping(mapping, 'file_name')

        assert cfg.EVAL.core_num >= 2

        run_times = 8  # batch中每一个，test 8次，从中找最好的
        for _ in range(run_times):
            for i in range(batch_size):
                kwargs = {}
                pass

                queue.put((i, batch_file_name[i], goals_scores_tuple[i], kwargs))

        while not queue.empty():
            pass

        expectations = np.ones(batch_size) * 10000.0
        batch_ans_points = np.zeros([batch_size, 6, 3])
        batch_pred_probs = np.zeros([batch_size, 6])
        for _ in range(run_times * batch_size):
            i, expectation, ans_points, pred_probs = queue_res.get()
            if expectation < expectations[i]:
                expectations[i] = expectation
                batch_ans_points[i, :, 0:2] = ans_points
                batch_pred_probs[i] = pred_probs

    # origin_point, origin_angle = batch_init_origin(mapping)

    # for i in range(batch_size):
    #     FDE = np.inf
    #     FDE = np.min(get_dis_point_2_points(batch_gt_points[i][-1], batch_ans_points[i]))
    #     method2FDE.append(FDE)
    #
    #     ans_points = batch_ans_points[i].copy()
    #     to_origin_coordinate(ans_points, i, origin_point, origin_angle)

    return batch_ans_points, batch_pred_probs, method2FDE
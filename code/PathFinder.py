import networkx as nx
from transformers import RobertaTokenizer
import signal
import time
import random

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

def count_token(path, code_dict, tokenizer):
    path_code = ''
    for line in path:
        path_code += code_dict[line]

    code_tokens = tokenizer.tokenize(path_code)
    token_num = len(code_tokens)
    return token_num

def random_select(all_paths, paths_num):
    if paths_num >= len(all_paths):
        return all_paths[:]
    else:
        return random.sample(all_paths, paths_num)

def greedy_select(all_paths, paths_num, code_dict, args):

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)
    covered = set()
    selected = []

    for _ in range(paths_num):
        best = None
        # best_gain = 0
        best_score = float('-inf')

        for path in all_paths:
            if path in selected:
                continue
            # gain = len(set(path) - covered)   # 新增覆盖的节点数

            new_nodes = set(path) - covered
            reward = args.alpha * len(new_nodes)
            path_token_num = count_token(path, code_dict, tokenizer)
            penalty = args.beta * max(0, path_token_num - 510)
            score = reward - penalty
            if score > best_score:
                best_score = score
                best = path

        if best is None:
            break
        selected.append(best)
        covered |= set(best)

    return selected

# def find_path(CFG, exit_nodes, code_dict, args):
#
#     all_paths = []
#     for tgt in exit_nodes:
#         # 所有不含环的简单路径
#         paths = list(nx.all_simple_paths(CFG, source=1, target=tgt))
#         all_paths.extend(paths)
#
#     rep_paths = greedy_select(all_paths, args.PathNum, 1.0, 0.2, code_dict, args)
#     if len(rep_paths) < args.PathNum:
#         all_nodes = list(CFG.nodes)
#         # print(all_nodes)
#
#         # 将所有节点视为一条路径，重复添加到 rep_paths 中，直到达到 3 条路径
#         while len(rep_paths) < args.PathNum:
#             rep_paths.append(all_nodes)
#
#     return rep_paths
#
# def find_path2(CFG, exit_nodes, code_dict, args):
#     all_paths = []
#     for tgt in exit_nodes:
#         # 所有不含环的简单路径
#         paths = list(nx.all_simple_paths(CFG, source=1, target=tgt))
#         all_paths.extend(paths)
#
#     print(len(all_paths))

def simple_partial_paths(CFG, exit_nodes, max_paths_per_target=30):
    """对每个出口节点只找前 N 条最短路径"""
    result = []
    for tgt in exit_nodes:
        try:
            # 使用 shortest_simple_paths 只取前 N 条
            gen = nx.shortest_simple_paths(CFG, source=1, target=tgt)
            for _, path in zip(range(max_paths_per_target), gen):
                result.append(path)
        except nx.NetworkXNoPath:
            pass

    return result

def count_paths(CFG, exit_nodes, time_limit):
    paths_nums_interval = 0

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(time_limit)  # 启动定时器
    all_paths = []
    simple_select = False
    try:
        for tgt in exit_nodes:
            paths = list(nx.all_simple_paths(CFG, source=1, target=tgt))
            all_paths.extend(paths)
        signal.alarm(0)  # 关闭定时器
    except TimeoutException:
        signal.alarm(0)
        simple_select = True
        all_paths = simple_partial_paths(CFG, exit_nodes)
        paths_nums_interval = 5
    """统计路径数量区间"""
    if not simple_select and len(all_paths) < 10:
        paths_nums_interval = 0
    elif not simple_select and len(all_paths) < 100:
        paths_nums_interval = 1
    elif not simple_select and len(all_paths) < 1000:
        paths_nums_interval = 2
    elif not simple_select and len(all_paths) < 10000:
        paths_nums_interval = 3
    elif not simple_select and len(all_paths) < 100000:
        paths_nums_interval = 4
    elif not simple_select:
        paths_nums_interval = 5

    return paths_nums_interval

def count_paths_num(CFG, exit_nodes, time_limit):

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(time_limit)  # 启动定时器
    all_paths = []
    try:
        for tgt in exit_nodes:
            paths = list(nx.all_simple_paths(CFG, source=1, target=tgt))
            all_paths.extend(paths)
        signal.alarm(0)  # 关闭定时器
    except TimeoutException:
        signal.alarm(0)
        all_paths = simple_partial_paths(CFG, exit_nodes)

    return len(all_paths)


def find_paths_with_timeout(CFG, exit_nodes, code_dict, args, time_limit):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(time_limit)  # 启动定时器
    all_paths = []
    select_status = 0
    simple_select = False
    try:
        for tgt in exit_nodes:
            paths = list(nx.all_simple_paths(CFG, source=1, target=tgt))
            all_paths.extend(paths)

        signal.alarm(0)  # 关闭定时器
    except TimeoutException:
        signal.alarm(0)
        simple_select = True
        # print(f"超过 {time_limit} 秒，切换到简化搜索策略")
        all_paths = simple_partial_paths(CFG, exit_nodes)
        select_status = 1
        paths_nums_interval = 4
        # print(len(all_paths))

    if not simple_select and len(all_paths) > 3000:
        # print(f"超过3000条，切换到简化搜索策略")
        all_paths = simple_partial_paths(CFG, exit_nodes)
        select_status = 2
        # print(len(all_paths))

    # print(len(all_paths))
    rep_paths = greedy_select(all_paths, args.PathNum, code_dict, args)
    # rep_paths = random_select(all_paths, args.PathNum)
    if len(rep_paths) < args.PathNum:
        all_nodes = list(CFG.nodes)
        # print(all_nodes)

        # 将所有节点视为一条路径，重复添加到 rep_paths 中，直到达到 3 条路径
        while len(rep_paths) < args.PathNum:
            rep_paths.append(all_nodes)

    return rep_paths, select_status
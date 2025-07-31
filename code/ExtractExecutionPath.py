import shutup; shutup.please()

import sys
sys.path.append('/home/user/PASMG/')
import argparse
from parserTool.utils import remove_comments_and_docstrings
import json
from ConstructASCFG import ConstructCFG, LabelNotFoundError, CommentNodeTypeError, ToMuchFunctionDefinitionError
# from ConstructCFG import ConstructCFG, LabelNotFoundError, CommentNodeTypeError, ToMuchFunctionDefinitionError
# from Construct_SCFG import ConstructCFG, LabelNotFoundError, CommentNodeTypeError, ToMuchFunctionDefinitionError
# from Construct_ACFG import ConstructCFG, LabelNotFoundError, CommentNodeTypeError, ToMuchFunctionDefinitionError
import parserTool.parse as ps
from parserTool.parse import Lang
from tqdm import tqdm
import re
import subprocess
import time
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
# from PathFinder import find_paths_with_timeout, count_paths, count_paths_num
# from PathFinder_Random import find_paths_with_timeout, count_paths, count_paths_num
from PathFinder_Shortest import find_paths_with_timeout, count_paths, count_paths_num
import signal
import os
from transformers import RobertaTokenizer


# def timeout_handler(signum, frame):
#     raise TimeoutError("The operation timed out.")

def extract_path(code_dict, path_sequence):
    paths = []
    for path in path_sequence:
        path_code = ''
        for line in path:
            if line in code_dict:
                path_code += code_dict[line]
        paths.append(path_code)

    return paths

# def remove_unnecessary_if_statement(code_dict, path_sequence):
#     for path in path_sequence:
#         lines_to_remove = []
#         for line in path:
#             line_code = code_dict[line]
#             # print("line code", line_code)
#             stripped_line_code = line_code.lstrip(' \t')
#             if stripped_line_code.startswith("if ") or stripped_line_code.startswith("if(") or stripped_line_code == "if":
#                 lines_to_remove.append(line)
#
#         # print(lines_to_remove)
#         # for i in lines_to_remove:
#         #     print(code_dict[i])
#
#         for line in lines_to_remove:
#             while line in path:
#                 path.remove(line)
#
#     return path_sequence

def count_lines(inputfile):
    with open(inputfile, 'r') as f:
        return sum(1 for _ in f)

def fix_for_statement(code):
    pattern = r'for\s*\(\s*([^\)]*)\s*\)\s*{'
    code = re.sub(pattern, lambda match: "for (" + match.group(1).replace("\n", " ") + ") {", code)

    # pattern = re.compile(r'(for\s*\(.*?\))\s*\n\s*;', re.DOTALL)
    # new_content = pattern.sub(r'\1;', code)
    # return new_content

    return code

def merge_multiline_statements(code):
    lines = code.split('\n')
    processed_lines = []
    current_buffer = []
    paren_count = 0
    operators = {',', '+', '-', '*', '/', '&', '|', '%', '<', '>', '=', '!', '^', '\\'}

    for line in lines:
        # stripped_line = line.strip()
        stripped_line = line
        if not current_buffer:
            current_buffer.append(stripped_line)
            paren_count += stripped_line.count('(') - stripped_line.count(')')
            merged = ' '.join(current_buffer)
            last_char = merged.strip()[-1] if merged.strip() else ''
            if paren_count > 0 or last_char in operators:
                continue
            else:
                processed_lines.append(merged)
                current_buffer = []
                paren_count = 0
        else:
            current_buffer.append(stripped_line)
            paren_count += stripped_line.count('(') - stripped_line.count(')')
            merged = ' '.join(current_buffer)
            stripped_merged = merged.strip()
            last_char = stripped_merged[-1] if stripped_merged else ''
            if paren_count > 0 or last_char in operators:
                continue
            else:
                processed_lines.append(merged)
                current_buffer = []
                paren_count = 0
    if current_buffer:
        merged = ' '.join(current_buffer)
        processed_lines.append(merged)
    return '\n'.join(processed_lines)

# def remove_blank_lines(code):
#     """
#     删除代码中的空行
#
#     参数:
#     code (str): 包含多行代码的字符串
#
#     返回:
#     str: 移除了空行后的代码字符串
#     """
#     # 将代码按行分割成列表
#     lines = code.splitlines()
#
#     # 使用列表推导式过滤掉空行或只包含空白字符的行
#     non_empty_lines = [line for line in lines if line.strip()]
#
#     # 将过滤后的行重新组合成一个字符串，并用换行符连接
#     cleaned_code = '\n'.join(non_empty_lines)
#
#     return cleaned_code

# def clang_format_code(code, use_config_file):
#     """
#     使用 clang-format 格式化代码，并将 for 循环压缩为单行。
#
#     :param code: 待格式化的代码字符串
#     :return: 格式化后的代码字符串
#     """
#     try:
#         style = "file" if use_config_file else "LLVM"
#         result = subprocess.run(
#             ["clang-format", f"-style={style}"],  # 指定格式化风格
#             input=code,  # 输入代码
#             text=True,  # 输入和输出均为文本
#             capture_output=True,  # 捕获标准输出
#             check=True  # 检查命令是否成功执行
#         )
#         return result.stdout.strip()  # 返回格式化后的代码
#     except subprocess.CalledProcessError as e:
#         print("格式化失败:", e)
#         return None

def code_to_dict(code_str):
    lines = code_str.splitlines()
    empty_lines = []
    result = {}

    for i, line in enumerate(lines):
        # processed_line = line.strip()
        if not line:
            empty_lines.append(i + 1)
        result[i + 1] = line

    if empty_lines:
        raise ValueError(f"代码中存在空行，请检查以下行号: {empty_lines}")

    return result

def fix_braces(code_str):
    lines = code_str.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped in ('{', '}'):
            if result:
                # 将大括号合并到前一行末尾
                result[-1] += stripped
            else:
                # 无法合并到前一行，直接添加原行
                result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)

# def split_goto_statements(code):
#     # 使用正则表达式匹配所有前面有空白字符的goto语句，并将其替换为换行后的形式
#     pattern = r'(\s+)(goto\s+\w+\s*;)'
#     # 替换为换行符加上goto语句，确保goto单独成行
#     new_code = re.sub(pattern, r'\n\2', code)
#     return new_code

def traverse(node):
    print(f"Node type: {node.type}, Text: {node.text}")
    for child in node.children:
        traverse(child)

def draw_directed_graph(G, node_color='lightblue', edge_color='gray', node_size=300, layout='spring'):
    """
    绘制有向图并显示节点编号

    参数:
    G (nx.DiGraph) - networkx有向图对象
    node_color (str) - 节点颜色
    edge_color (str) - 边颜色
    node_size (int) - 节点大小
    layout (str) - 布局方式 ('spring' 或 'circular')
    """
    plt.figure(figsize=(16, 16))

    # 选择布局
    if layout == 'circular':
        pos = nx.circular_layout(G)
    else:  # 默认使用spring布局
        pos = nx.spring_layout(G)

    # 绘制元素
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_labels(G, pos)  # 显示节点编号
    nx.draw_networkx_edges(G, pos,
                           arrowstyle='->',
                           arrowsize=5,
                           edge_color=edge_color,
                           connectionstyle='arc3,rad=0.7')  # 添加弧线避免重叠

    plt.axis('off')
    plt.savefig("./1.png", dpi=300, bbox_inches='tight')
    plt.show()

def level_order_traversal_with_level(root):
    """
    层次遍历一棵树，并打印每个节点的类型及其所在层数。

    :param root: 树的根节点 (TreeNode 类型)
    """
    if not root:
        print("树为空")
        return

    queue = deque([(root, 0)])  # 使用队列进行层次遍历，元组 (node, level) 表示节点和其所在层数

    while queue:
        current_node, level = queue.popleft()  # 取出当前节点和其所在层数
        # print(current_node)
        print(f"第 {level} 层，节点类型: {current_node.type},开始于：{current_node.start_point},结束于：{current_node.end_point}")  # 打印节点类型及层数

        # 将当前节点的子节点加入队列，并将层数加 1
        for child in current_node.children:
            queue.append((child, level + 1))

def can_reach_all_exits(graph, start_node, exit_nodes):
    """
    判断从start_node是否可以到达有向图graph中的所有指定出口节点。

    :param graph: networkx.DiGraph 对象
    :param start_node: 起始节点
    :param exit_nodes: 需要到达的出口节点集合
    :return: 如果可以从start_node到达所有出口节点，则返回True；否则返回False
    """
    # 将出口节点转换为集合，并处理空集合的情况
    remaining_exits = set(exit_nodes)
    if not remaining_exits:
        return True  # 没有需要到达的出口节点，视为可达

    # 可选：检查出口节点是否都在图中存在
    # if not remaining_exits.issubset(graph.nodes):
    #     return False

    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            # 如果当前节点是未访问的出口节点，更新剩余出口节点
            if node in remaining_exits:
                remaining_exits.remove(node)
                # 所有出口节点已访问，提前返回
                if not remaining_exits:
                    return True
            # 将邻居节点加入栈中
            stack.extend(graph.neighbors(node))

    # 遍历结束后检查是否所有出口节点都被访问
    return not remaining_exits

# def coverage_percentage(L, k):
#     # 将所有子列表中的元素合并到一个集合中，自动去重
#     all_numbers = set()
#     for sublist in L:
#         all_numbers.update(sublist)
#
#     # 生成1到k的整数集合
#     required_numbers = set(range(1, k + 1))
#
#     # 计算覆盖的数量
#     covered_count = len(all_numbers & required_numbers)
#
#     # 计算百分比
#     return (covered_count / k) * 100

def coverage_percentage(L, k, i=None):
    if i is None:
        selected_lists = L
    else:
        selected_lists = L[:i+1]

    all_numbers = set()
    for sublist in selected_lists:
        all_numbers.update(sublist)

    required_numbers = set(range(1, k + 1))
    covered_numbers = all_numbers & required_numbers
    covered_count = len(covered_numbers)

    return (covered_count / k) * 100

def remove_blank(code):
    lines = code.splitlines()
    trimmed_lines = [line.strip() for line in lines]

    return '\n'.join(trimmed_lines)

def preprocessing_ablation(inputfile1, inputfile2, inputfile3, args):
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)

    # 定义不同预处理阶段的步骤组合
    processing_stages = [
        {"name": "step_1", "steps": []},  # 只移除注释
        {"name": "step_2", "steps": [remove_blank]},  # 再删除缩进
        {"name": "step_3", "steps": [remove_blank, fix_for_statement, merge_multiline_statements]},  # 再合并多行
        {"name": "step_4", "steps": [remove_blank, fix_for_statement, merge_multiline_statements, fix_braces]}  # 再提大括号
    ]

    # 统一处理所有输入文件
    input_files = [inputfile1, inputfile2, inputfile3]

    for stage in processing_stages:
        total_tokens = 0
        total_lines = 0
        sample_count = 0

        for file_path in input_files:
            with open(file_path, 'r') as f:
                for line in f:
                    js = json.loads(line.strip())
                    code = js['func']

                    # 基础处理：移除注释
                    clean_code, _ = remove_comments_and_docstrings(code, 'c')

                    # 应用当前阶段的额外处理
                    for processing_step in stage["steps"]:
                        clean_code = processing_step(clean_code)

                    # print(clean_code)

                    # 统计指标
                    total_tokens += len(tokenizer.tokenize(clean_code))
                    total_lines += len(clean_code.splitlines())
                    # print(len(clean_code.splitlines()))
                    sample_count += 1

        # 输出当前阶段的结果
        print(f"[{stage['name']}] Average tokens: {total_tokens / sample_count:.2f}")
        print(f"[{stage['name']}] Average lines: {total_lines / sample_count:.2f}")
        print("-------------------------------------------------------------")

def cdf(inputfile1, inputfile2, inputfile3):
    input_files = [inputfile1, inputfile2, inputfile3]
    output_file = "ASCFG_paths_nums.txt"

    for file_path in input_files:
        with open(file_path, 'r') as f:
            total_lines = count_lines(file_path)
            for line in tqdm(f, desc="Reading lines", total=total_lines):
            # for line in f:
                js = json.loads(line.strip())
                code = js['func']
                clean_code, _ = remove_comments_and_docstrings(code, 'c')
                clean_code = remove_blank(clean_code)
                clean_code = fix_for_statement(clean_code)
                clean_code = merge_multiline_statements(clean_code)
                clean_code = fix_braces(clean_code)
                code_ast = ps.tree_sitter_ast(clean_code, Lang.C)

                try:
                    CFG, exit_nodes = ConstructCFG(code_ast.root_node, clean_code)
                    paths_nums = count_paths_num(CFG, exit_nodes, 5)
                    if paths_nums !=0 :
                        with open(output_file, 'a') as out_f:  # 使用 'a' 模式追加写入
                            out_f.write(f"{paths_nums}\n")  # 每个数字一行

                except (LabelNotFoundError, CommentNodeTypeError,
                        ToMuchFunctionDefinitionError, RecursionError, OSError):
                    continue

    # line_counts = [x for x in line_counts if x != 0]
    # sorted_counts = np.sort(line_counts)
    # n = len(sorted_counts)
    # y = np.arange(1, n + 1) / n
    # y_percent = y * 100
    #
    # # 转换为 log 空间
    # x_log = np.log10(sorted_counts)
    # x_log_smooth = np.linspace(x_log.min(), x_log.max(), 500)
    # spline = UnivariateSpline(x_log, y_percent, k=2, s=0)  # s=0 表示严格插值
    # y_smooth = spline(x_log_smooth)
    # x_smooth = 10 ** x_log_smooth  # 还原 log 空间
    #
    # # 创建图表
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_smooth, y_smooth, color='#1f77b4', linewidth=2.5, label='Smoothed CDF')
    #
    # # 设置对数坐标轴
    # plt.xscale('log')
    #
    # # 设置轴标签和标题
    # plt.xlabel('Number of Simple Paths (log scale)', fontsize=12, labelpad=10)
    # plt.ylabel('Cumulative Percentage (%)', fontsize=12, labelpad=10)
    # plt.title('Cumulative Distribution of Simple Paths per Function',
    #           fontsize=14, pad=15)
    #
    # # 优化刻度标签
    # plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    # plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
    # plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    #
    # # 添加网格线
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    #
    # # 设置坐标范围
    # plt.xlim(left=0.9)  # 为避免 log(0) 错误，略小于1
    # plt.ylim(bottom=0, top=100)
    #
    # # 添加关键百分位标记（仍使用原始数据）
    # percentiles = [50, 75, 90, 95, 99]
    # for p in percentiles:
    #     idx = np.searchsorted(y, p / 100, side='left')
    #     if idx < n:
    #         x_val = sorted_counts[idx]
    #         y_val = y_percent[idx]
    #         plt.scatter(x_val, y_val, s=60, zorder=5,
    #                     edgecolor='#ff7f0e', facecolor='white', linewidth=1.5)
    #         plt.text(x_val * 1.3, y_val - 3, f'P{p}: {x_val}\n{y_val:.1f}%',
    #                  fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    #
    # # 添加图例和数据统计
    # stats_text = f"Total Samples: {n}\nMin: {sorted_counts[0]}\nMax: {sorted_counts[-1]}"
    # plt.annotate(stats_text, xy=(0.98, 0.02), xycoords='axes fraction',
    #              fontsize=10, ha='right', va='bottom',
    #              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #
    # plt.legend(loc='lower right', fontsize=10)
    # plt.tight_layout()
    #
    # # 保存图片
    # output_path = 'path_count_cdf.png'
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"CDF plot saved to {output_path}")


def display_path_distribution(inputfile):
    total_lines = count_lines(inputfile)
    with open(inputfile, 'r') as f:
        label_not_found_count = 0
        comment_node_type_error_count = 0
        too_much_function_definition_error_count = 0
        recursion_error_count = 0
        os_error_count = 0
        unreachable_exit_count = 0

        count_0_to_9 = 0
        count_10_to_99 = 0
        count_100_to_999 = 0
        count_1000_to_9999 = 0
        count_10000_to_99999 = 0
        count_100000_plus = 0

        for line in tqdm(f, total=total_lines, desc='Counting Execution Path'):
            line = line.strip()
            js = json.loads(line)

            #remove_blank, fix_for_statement, merge_multiline_statements, fix_braces

            clean_code, _ = remove_comments_and_docstrings(js['func'], 'c')
            clean_code = remove_blank(clean_code)
            clean_code = fix_for_statement(clean_code)
            clean_code = merge_multiline_statements(clean_code)
            clean_code = fix_braces(clean_code)

            # clean_code = fix_for_statement(clean_code)
            # clean_code = merge_multiline_statements(clean_code)
            # clean_code = fix_braces(clean_code)

            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)

            try:
                CFG, exit_nodes = ConstructCFG(code_ast.root_node, clean_code)
                paths_nums_interval = count_paths(CFG, exit_nodes, 5)
                if paths_nums_interval == 0:
                    count_0_to_9 += 1
                elif paths_nums_interval == 1:
                    count_10_to_99 += 1
                elif paths_nums_interval == 2:
                    count_100_to_999 += 1
                elif paths_nums_interval == 3:
                    count_1000_to_9999 += 1
                elif paths_nums_interval == 4:
                    count_10000_to_99999 += 1
                elif paths_nums_interval == 5:
                    count_100000_plus += 1

            except LabelNotFoundError:
                label_not_found_count += 1
                continue
            except CommentNodeTypeError:
                comment_node_type_error_count += 1
                continue
            except ToMuchFunctionDefinitionError:
                too_much_function_definition_error_count += 1
                continue
            except RecursionError:
                recursion_error_count += 1
                continue
            except OSError:
                os_error_count += 1
                continue

            if can_reach_all_exits(CFG, 1, exit_nodes):
                pass
            else:
                unreachable_exit_count += 1
                continue

        print(f"LabelNotFoundError occurred {label_not_found_count} times.")
        print(f"CommentNodeTypeError occurred {comment_node_type_error_count} times.")
        print(f"TooMuchFunctionDefinitionError occurred {too_much_function_definition_error_count} times.")
        print(f"RecursionError occurred {recursion_error_count} times.")
        print(f"OSError occurred {os_error_count} times.")
        print(f"Number of CFGs that cannot reach all exit nodes: {unreachable_exit_count}")

        print("Path Count Distribution in the Dataset:")

        print(f"Samples with 0-9 paths: {count_0_to_9}")
        print(f"Samples with 10-99 paths: {count_10_to_99}")
        print(f"Samples with 100-999 paths: {count_100_to_999}")
        print(f"Samples with 1000-9999 paths: {count_1000_to_9999}")
        print(f"Samples with 10000-99999 paths: {count_10000_to_99999}")
        print(f"Samples with 100000+ paths: {count_100000_plus}")

def func(inputfile, outputfile, args):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)
    total_lines = count_lines(inputfile)
    with open(inputfile, 'r') as f, open(outputfile, 'w') as w:
        for line in tqdm(f, total=total_lines, desc='Extracting Execution Path'):
            line = line.strip()
            js = json.loads(line)
            clean_code, _ = remove_comments_and_docstrings(js['func'], 'c')
            clean_code = remove_blank(clean_code)
            clean_code = fix_for_statement(clean_code)
            clean_code = merge_multiline_statements(clean_code)
            clean_code = fix_braces(clean_code)

            new_entry = {}
            new_entry["path1"] = clean_code
            new_entry["path2"] = clean_code
            new_entry["path3"] = clean_code
            new_entry["path4"] = clean_code
            new_entry["target"] = js['target']
            w.write(json.dumps(new_entry) + '\n')


def ExtractExecutionPath(inputfile, outputfile, args):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)
    total_lines = count_lines(inputfile)
    with open(inputfile, 'r') as f, open(outputfile, 'w') as w:
        label_not_found_count = 0
        comment_node_type_error_count = 0
        too_much_function_definition_error_count = 0
        recursion_error_count = 0
        os_error_count = 0
        unreachable_exit_count = 0
        timeout_rollback = 0
        explosion_rollback = 0

        coverage_sum = [0] * args.PathNum
        coverage_count = 0

        for line in tqdm(f, total=total_lines, desc='Extracting Execution Path'):
            line = line.strip()
            js = json.loads(line)

            # clean_code, _ = remove_comments_and_docstrings(js['func'], 'c') #clean_code为去除注释的代码 code_dict无用
            # clean_code = fix_for_statement(clean_code)
            # clean_code = merge_multiline_statements(clean_code)
            # clean_code = fix_braces(clean_code)

            clean_code, _ = remove_comments_and_docstrings(js['func'], 'c')
            clean_code = remove_blank(clean_code)
            clean_code = fix_for_statement(clean_code)
            clean_code = merge_multiline_statements(clean_code)
            clean_code = fix_braces(clean_code)

            # after = remove_blank_lines(clean_code)
            # if after == clean_code:
            #     print("!!!!!!!!!!!!!!!!!")
            # clean_code = split_goto_statements(clean_code)
            code_dict = code_to_dict(clean_code)

            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)

            # print(clean_code)

            level_order_traversal_with_level(code_ast.root_node)
            # time.sleep(1000)


            try:
                CFG, exit_nodes = ConstructCFG(code_ast.root_node, clean_code)
                # draw_directed_graph(CFG)
                # time.sleep(1000)
                # signal.signal(signal.SIGALRM, timeout_handler)
                # signal.alarm(30)  # 10秒后触发 SIGALRM 信号
                # find_path2(CFG, exit_nodes, code_dict, args)
                #
                paths, select_status = find_paths_with_timeout(CFG, exit_nodes, code_dict, args, 5)
                # print(paths)
                if select_status == 1:
                    timeout_rollback += 1
                elif select_status == 2:
                    explosion_rollback += 1

                for i in range(args.PathNum):
                    coverage_sum[i] += coverage_percentage(paths, len(code_dict), i)
                coverage_count += 1
                # paths = find_path(CFG, exit_nodes, code_dict, args)
                path_code = extract_path(code_dict, paths)

            except LabelNotFoundError:
                label_not_found_count += 1
                continue
            except CommentNodeTypeError:
                comment_node_type_error_count += 1
                continue
            except ToMuchFunctionDefinitionError:
                too_much_function_definition_error_count += 1
                continue
            # except TimeoutError:
            #     print("外层超时")
            #     timeout_error_count += 1
            #     continue
            except RecursionError:
                recursion_error_count += 1
                continue
            except OSError:
                os_error_count += 1
                continue


            if can_reach_all_exits(CFG, 1, exit_nodes):
                pass
            else:
                # print("不能到达所有出口节点")
                # draw_directed_graph(CFG)
                # ruduwei0 = [node for node in CFG.nodes if CFG.in_degree(node) == 0]
                # print(ruduwei0)
                # print(js['func'])
                # print(clean_code)
                unreachable_exit_count += 1
                # time.sleep(4)
                continue

            new_entry = {}
            for i, path in enumerate(path_code, start=1):
                new_entry[f"path{i}"] = path
            new_entry["target"] = js['target']
            w.write(json.dumps(new_entry) + '\n')

        print(f"LabelNotFoundError occurred {label_not_found_count} times.")
        print(f"CommentNodeTypeError occurred {comment_node_type_error_count} times.")
        print(f"TooMuchFunctionDefinitionError occurred {too_much_function_definition_error_count} times.")
        print(f"RecursionError occurred {recursion_error_count} times.")
        print(f"OSError occurred {os_error_count} times.")
        print(f"Number of CFGs that cannot reach all exit nodes: {unreachable_exit_count}")
        print(f"Timeout rollback occurred {timeout_rollback} times.")
        print(f"Explosion rollback occurred {explosion_rollback} times.")
        for path_num in range(args.PathNum):
            print(f"When there are {path_num + 1} paths, the average CFG node coverage is {coverage_sum[path_num] / coverage_count:.2f}%")
        print("-------------------------------------------------------------")

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--training_set", default="../Dataset/FuncDataset/Devign/train.jsonl", type=str)
    parser.add_argument("--validation_set", default="../Dataset/FuncDataset/Devign/valid.jsonl", type=str)
    parser.add_argument("--test_set", default="../Dataset/FuncDataset/Devign/test.jsonl", type=str)

    parser.add_argument("--ExecutionPaths_training_set", default="../Dataset/4PathsDataset/14/Devign/train.jsonl", type=str)
    parser.add_argument("--ExecutionPaths_validation_set", default="../Dataset/4PathsDataset/14/Devign/valid.jsonl", type=str)
    parser.add_argument("--ExecutionPaths_test_set", default="../Dataset/4PathsDataset/14/Devign/test.jsonl", type=str)

    parser.add_argument("--pretrained_model_path", default="../pretrain-model/pdbert", type=str)

    parser.add_argument("--PathNum", default=4, type=int)
    parser.add_argument("--alpha", default=14, type=int)
    parser.add_argument("--beta", default=1, type=int)
    args = parser.parse_args()

    ExtractExecutionPath(args.training_set, args.ExecutionPaths_training_set, args)
    ExtractExecutionPath(args.validation_set, args.ExecutionPaths_validation_set, args)
    ExtractExecutionPath(args.test_set, args.ExecutionPaths_test_set, args)

    # func(args.training_set, args.ExecutionPaths_training_set, args)
    # func(args.validation_set, args.ExecutionPaths_validation_set, args)
    # func(args.test_set, args.ExecutionPaths_test_set, args)

    # display_path_distribution(args.training_set)
    # display_path_distribution(args.validation_set)
    # display_path_distribution(args.test_set)

    # preprocessing_ablation(args.training_set, args.validation_set, args.test_set, args)

    # cdf(args.training_set, args.validation_set, args.test_set)

if __name__ == "__main__":
    main()
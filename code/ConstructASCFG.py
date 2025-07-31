import ast
import networkx as nx
from itertools import islice

CFG = None
exit_nodes = []
exist_return = False
goto_list = []
labeled_list = []
code = None

class LabelNotFoundError(Exception):
    pass
class CommentNodeTypeError(Exception):
    pass
class ToMuchFunctionDefinitionError(Exception):
    pass


def safe_remove_edge(graph, u, v):
    """静默移除边（如果存在），无异常抛出"""
    try:
        graph.remove_edge(u, v)
    except nx.NetworkXError:  # NetworkX 移除不存在边时抛出的异常类型
        pass

def extract_text(start_row, start_col, end_row, end_col):
    global code

    lines = code.splitlines(True)
    if not lines:
        return ""

    result = []
    for i in range(start_row, end_row + 1):
        line = lines[i]
        line_len = len(line)

        if i == start_row and i == end_row:
            s = max(0, min(start_col, line_len))
            e = max(s, min(end_col, line_len))
        elif i == start_row:
            s = max(0, min(start_col, line_len))
            e = line_len
        elif i == end_row:
            s = 0
            e = max(0, min(end_col, line_len))
        else:
            s = 0
            e = line_len

        result.append(line[s:e])

    return "".join(result)

def handle_simple_leaf_node(node):
    global CFG
    for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
        CFG.add_edge(i, i + 1, weight=1)

def handle_goto_statement(node):
    global CFG, goto_list
    for child_node in node.children:
        if child_node.type == 'statement_identifier':
            label_name = extract_text(child_node.start_point[0], child_node.start_point[1], child_node.end_point[0], child_node.end_point[1])
            label = {'name': label_name, 'line': child_node.start_point[0]+1}
            goto_list.append(label)

def handle_labeled_statement(node):
    global CFG, labeled_list
    for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
        CFG.add_edge(i, i + 1, weight=1)

    for child_node in node.children:
        if child_node.type == 'statement_identifier':
            label_name = extract_text(child_node.start_point[0], child_node.start_point[1], child_node.end_point[0], child_node.end_point[1])
            label = {'name': label_name, 'line': child_node.start_point[0]+1}
            labeled_list.append(label)
        traverse_tree(child_node)

def handle_if_statement(node):
    global CFG
    exist_else_clause = False

    for child_node in node.children:
        if child_node.type == "else_clause":
            exist_else_clause = True
    if exist_else_clause:
        for child_node in node.children:
            if child_node.type == "else_clause":
                else_clause_start_point = child_node.start_point[0] + 1
                # 找到所有指向节点 B 的节点及其边的权重
                for source, _ in CFG.in_edges(node.start_point[0] + 1):
                    # 获取原边的权重
                    weight = CFG[source][node.start_point[0] + 1]['weight']
                    # 添加从 source 到 E 的边，并保持相同的权重
                    CFG.add_edge(source, child_node.start_point[0] + 1, weight=weight)
    else:
        for source, _ in CFG.in_edges(node.start_point[0] + 1):
            # 获取原边的权重
            weight = CFG[source][node.start_point[0] + 1]['weight']
            # 添加从 source 到 E 的边，并保持相同的权重
            CFG.add_edge(source, node.end_point[0] + 2, weight=weight)
    for child_node in node.children:
        traverse_tree(child_node)
    if exist_else_clause:
        for child_node in node.children:
            if child_node.type == "compound_statement":
                if child_node.end_point[0] + 1 == else_clause_start_point:
                    # CFG.remove_edge(child_node.end_point[0], child_node.end_point[0] + 1)
                    safe_remove_edge(CFG, child_node.end_point[0], child_node.end_point[0] + 1)
                    CFG.add_edge(child_node.end_point[0], node.end_point[0] + 2, weight=1)
                else:
                    CFG.add_edge(child_node.end_point[0] + 1, node.end_point[0] + 2, weight=1)
                    # CFG.remove_edge(child_node.end_point[0] + 1, child_node.end_point[0] + 2)
                    safe_remove_edge(CFG, child_node.end_point[0] + 1, child_node.end_point[0] + 2)

def handle_pass_node(node):
    pass

def handle_simple_branch_node(node):
    for child_node in node.children:
        traverse_tree(child_node)

def handle_translation_unit(node):
    function_definition_num = 0
    for index, root_child in enumerate(node.children):

        if root_child.type == "function_definition":
            function_definition_num += 1
            if function_definition_num > 1:
                # raise SkipIteration
                raise ToMuchFunctionDefinitionError("More than one function definition found.")

        traverse_tree(root_child)

def handle_return_statement(node):
    global CFG, exit_nodes, exist_return
    exist_return = True
    # print("use")

    exit_nodes.append(node.end_point[0] + 1)
    for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
        CFG.add_edge(i, i + 1, weight=1)

default_handlers = {
    "translation_unit": handle_translation_unit,
#------------------------------------------------------------
    "compound_statement": handle_simple_branch_node,
    "else_clause": handle_simple_branch_node,
    "function_definition": handle_simple_branch_node,
#------------------------------------------------------------
    "attribute_specifier": handle_simple_leaf_node,
    "break_statement": handle_simple_leaf_node,  #后续如果要编写对switch语句的分支，则需要编写break_statement处理逻辑
    "case_statement": handle_simple_leaf_node,
    "class_declaration": handle_simple_leaf_node,
    "continue_statement": handle_simple_leaf_node,
    "declaration": handle_simple_leaf_node,
    "do_statement": handle_simple_leaf_node,
    "else": handle_simple_leaf_node,
    "enum_specifier": handle_simple_leaf_node,
    "expression_statement": handle_simple_leaf_node,
    "for_statement": handle_simple_leaf_node,
    "function_declarator": handle_simple_leaf_node,
    "identifier": handle_simple_leaf_node,
    "if": handle_simple_leaf_node,
    "macro_type_specifier": handle_simple_leaf_node,
    "ms_call_modifier": handle_simple_leaf_node,
    "parenthesized_expression": handle_simple_leaf_node,
    "parenthesized_declarator": handle_simple_leaf_node,
    "pointer_declarator": handle_simple_leaf_node,
    "preproc_call": handle_simple_leaf_node,
    "preproc_def": handle_simple_leaf_node,
    "preproc_defined": handle_simple_leaf_node,
    "preproc_else": handle_simple_leaf_node,
    "preproc_function_def": handle_simple_leaf_node,
    "preproc_if": handle_simple_leaf_node,
    "preproc_ifdef": handle_simple_leaf_node,
    "preproc_include": handle_simple_leaf_node,
    "primitive_type": handle_simple_leaf_node,
    "sized_type_specifier": handle_simple_leaf_node,
    "statement_identifier": handle_simple_leaf_node,
    "storage_class_specifier": handle_simple_leaf_node,
    "struct_specifier": handle_simple_leaf_node,
    "switch_statement": handle_simple_leaf_node,
    "type_definition": handle_simple_leaf_node,
    "type_identifier": handle_simple_leaf_node,
    "type_qualifier": handle_simple_leaf_node,
    "union_specifier": handle_simple_leaf_node,
    "while_statement": handle_simple_leaf_node,
    "ERROR": handle_simple_leaf_node,
    "{": handle_simple_leaf_node,
    "}": handle_simple_leaf_node,
    ";": handle_simple_leaf_node,
    ":": handle_simple_leaf_node,
#------------------------------------------------------------
    "goto_statement": handle_goto_statement,
    "if_statement": handle_if_statement,
    "labeled_statement": handle_labeled_statement,
    "return_statement": handle_return_statement,
}

def traverse_tree(node, handlers=default_handlers):
    """
    遍历树，本质上是根据节点类型调用对应的处理函数。

    :param node: 节点 (TreeNode)
    :param handlers: 一个字典，键为节点类型，值为对应的处理函数
    """
    if node is None:
        return

    # 根据节点类型调用对应的处理函数
    if node.type in handlers:
        handlers[node.type](node)  # 调用处理函数
    elif node.type == "comment":
        raise CommentNodeTypeError("comment node type")
    else:
        print(f"No handler found for node type: {node.type}")

def find_goto_and_label(node):
    # 如果当前节点类型匹配，则执行操作
    if node.type == "goto_statement" or node.type == "labeled_statement":
        traverse_tree(node)

    # 递归处理子节点
    for child_node in node.children:
        find_goto_and_label(child_node)

def ConstructCFG(root_node, code_in):
    global CFG, exit_nodes, exist_return, goto_list, labeled_list, code
    CFG = nx.DiGraph()
    exit_nodes = []
    exist_return = False
    goto_list = []
    labeled_list = []
    code = code_in

    for i in range(root_node.start_point[0] + 1, root_node.end_point[0] + 2):
        CFG.add_node(i)
    traverse_tree(root_node)

    # CFG.remove_node(root_node.end_point[0] + 2) #因为在处理最后一行代码时会创建一个不存在的下一个节点
    if CFG.has_node(root_node.end_point[0] + 2):
        CFG.remove_node(root_node.end_point[0] + 2)

    if not exist_return:
        exit_nodes = [node for node in CFG.nodes if CFG.out_degree(node) == 0]
        # exit_nodes.append(root_node.end_point[0] + 1)

    find_goto_and_label(root_node)

    edges_to_remove = [(node, neighbor) for node in exit_nodes for neighbor in CFG.successors(node)]
    # goto_lines = {node['line'] for node in goto_list}
    # goto_edges_to_remove = [(line, neighbor) for line in goto_lines for neighbor in CFG.successors(line)]
    # edges_to_remove.extend(goto_edges_to_remove)
    CFG.remove_edges_from(edges_to_remove)


    # print(goto_list)
    # print(labeled_list)

    label_dict = {label['name']: label['line'] for label in labeled_list}
    for goto in goto_list:
        label_name = goto['name']
        i = goto['line']
        # 查找对应的标签行号
        j = label_dict.get(label_name)
        if j is not None:
            CFG.add_edge(i, j, weight=1)
        else:
            raise LabelNotFoundError(f"Label '{label_name}' not found for goto at line {i}")

    for label_line in label_dict.values():
        if CFG.in_degree(label_line) == 0:
            CFG.add_edge(label_line - 1, label_line, weight=1)

    return CFG, exit_nodes
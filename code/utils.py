import re
from collections import defaultdict, Counter
import heapq

def parse_cfg_edges(cfg_content):
    edges = []
    nodes = set()
    for line in cfg_content.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            nodes.add(u)
            nodes.add(v)
            edges.append((u, v))
    return edges, sorted(nodes)

def extract_loops(ptx_content):
    loops = {}
    active_loops_stack = []
    depth = 0

    for line in ptx_content.split('\n'):
        begin_match = re.search(r'//\s*LOOP_BEGIN,\s*(\d+),\s*(\d+)', line)
        end_match = re.search(r'//\s*LOOP_END,\s*(\d+)', line)

        if begin_match:
            loop_id = int(begin_match.group(1))
            bound = int(begin_match.group(2))
            depth = len(active_loops_stack) + 1
            loops[loop_id] = {'bound': bound, 'depth': depth}
            active_loops_stack.append(loop_id)

        if end_match:
            loop_id = int(end_match.group(1))
            if active_loops_stack and active_loops_stack[-1] == loop_id:
                active_loops_stack.pop()
                depth = len(active_loops_stack)
    sorted_loops = dict(sorted(loops.items(), key=lambda item: item[1]['depth']))
    return sorted_loops

def map_basic_blocks(bb_content):
    bb_instructions = defaultdict(list)
    bb_labels = {}

    current_bb = None
    current_bb_id = None

    for line in bb_content.split('\n'):
        if not line.strip():
            continue

        bb_match = re.match(r'(bb_(\d+))\s*:\s*(.*)', line)
        if bb_match:
            current_bb_id = int(bb_match.group(2))
            content = bb_match.group(3).strip()
            if content:
                bb_instructions[current_bb_id].append(content)

            if content.startswith('BB'):
                bb_labels[content.rstrip(';')] = current_bb_id

    return bb_instructions, bb_labels

def find_loop_blocks(ptx_content, bb_content, loops):
    loop_body = defaultdict(set)
    active_loops = []

    for line in ptx_content.split('\n'):
        begin_match = re.search(r'//\s*LOOP_BEGIN,\s*(\d+),\s*(\d+)', line)
        end_match = re.search(r'//\s*LOOP_END,\s*(\d+)', line)

        if begin_match:
            loop_id = int(begin_match.group(1))
            if loop_id in loops:
                active_loops.append(loop_id)
                loops[loop_id]['start_line'] = line

        if end_match:
            loop_id = int(end_match.group(1))
            if active_loops and active_loops[-1] == loop_id:
                loops[loop_id]['end_line'] = line
                active_loops.pop()

    bb_instructions, _ = map_basic_blocks(bb_content)

    loop_blocks = defaultdict(set)

    for loop_id, info in loops.items():
        if info['start_line'] is None or info['end_line'] is None:
            continue

        start_found = False
        end_found = False
        current_instrs = Counter()
        sum = 0

        for line in ptx_content.split('\n'):
            if line.strip() == info['start_line'].strip():
                start_found = True
                continue

            if line.strip() == info['end_line'].strip():
                end_found = True
                break

            if start_found and not end_found:
                if line.strip() and not line.strip().startswith('//'):
                    instr = re.sub(r'\s', '', line)
                    current_instrs[instr] += 1
                    sum += 1

        for block_id, instrs in bb_instructions.items():
            num = -1
            block_instrs = Counter()
            for instr in instrs:
                norm_instr = re.sub(r'\s', '', instr)
                if norm_instr.startswith('BB') or norm_instr.startswith('//'):
                    continue
                num += 1
                if norm_instr in current_instrs:
                    block_instrs[norm_instr] += 1
                    if block_instrs[norm_instr] > current_instrs[norm_instr]:
                        num = -1
                        break
                else:
                    num = -1
                    break

            if num != -1:
                loop_blocks[loop_id].add(block_id)

    return loop_blocks

def break_loops(edges, loop_blocks):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    removed_edges = set()

    for loop_id, blocks in loop_blocks.items():
        loop_head = min(blocks)

        for u, v in edges:
            if v == loop_head and u in blocks:
                if u in graph and v in graph[u]:
                    removed_edges.add((u, v))

    new_edges = []
    for u, v in edges:
        if (u, v) not in removed_edges:
            new_edges.append((u, v))

    return new_edges

def compute_block_counts(loops, loop_blocks, nodes):
    block_counts = {node: 1 for node in nodes}

    for loop_id, blocks in loop_blocks.items():
        bound = loops[loop_id]['bound']
        for block in blocks:
            block_counts[block] *= bound

    return block_counts

def topological_sort(edges):
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    for node in nodes:
        if node not in in_degree:
            in_degree[node] = 0

    heap = [node for node in in_degree if in_degree[node] == 0]
    heapq.heapify(heap)
    topo_order = []

    while heap:
        node = heapq.heappop(heap)
        topo_order.append(node)

        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(topo_order) != len(nodes):
        remaining = sorted(nodes - set(topo_order))
        topo_order.extend(remaining)

    return topo_order

def unroll_loops(path, loops, loop_blocks):
    loops_sorted = sorted(loops.items(), key=lambda x: x[1]['depth'])

    unrolled_path = path[:]
    path_marks = [None] * len(unrolled_path)

    for loop_id, loop_info in loops_sorted:
        if loop_id not in loop_blocks:
            continue

        loop_head = min(loop_blocks[loop_id])
        loop_body = loop_blocks[loop_id]
        bound = loop_info['bound']

        i = 0
        while i < len(unrolled_path):
            node = unrolled_path[i]
            if node == loop_head:
                j = i
                while j < len(unrolled_path) and unrolled_path[j] in loop_body:
                    j += 1

                segment = unrolled_path[i:j]
                segment_len = len(segment)

                insert_pos = j
                for _ in range(bound - 1):
                    unrolled_path[insert_pos:insert_pos] = segment
                    path_marks[insert_pos:insert_pos] = [loop_id] * segment_len
                    insert_pos += segment_len

                i = insert_pos
            else:
                i += 1

    return unrolled_path, path_marks

def nodepredecessor(edges, unrolled_path, path_marks, loop_blocks):
    pred_map = defaultdict(list)
    for u, v in edges:
        pred_map[v].append(u)

    latest_occurrence = {}
    latest_mark = {}

    predecessors = []

    for i, node in enumerate(unrolled_path):
        current_preds = []

        if i > 0:
            for p_node in pred_map.get(node, []):
                if p_node in latest_occurrence:
                    current_loop_id = path_marks[i]
                    if current_loop_id is not None and p_node not in loop_blocks.get(current_loop_id, set()):
                        continue

                    if p_node == node and latest_mark.get(p_node, None) is not None and path_marks[i] is not None:
                        if latest_mark[p_node] > path_marks[i]:
                            continue

                    current_preds.append(p_node)

        predecessors.append(current_preds)

        latest_occurrence[node] = i
        latest_mark[node] = path_marks[i]

    return predecessors
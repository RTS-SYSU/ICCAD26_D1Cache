import os
import sys
from collections import defaultdict

from gpu_config import GPUConfig
from multi_warp_analyse import MultiWarpWCET
from syc_warp_analyse import SycWarpWCET
from inst_warp_analyse import WarpMemoryAnalyzer
from utils import parse_cfg_edges, extract_loops, map_basic_blocks, find_loop_blocks, break_loops, compute_block_counts, \
    topological_sort, unroll_loops, nodepredecessor
from warp_allocate import Warp_Block_SM_Allocate
from ZW2018_analyse import ZW2018WarpWCET
from RTSS2025_analyse import RTSSWarpWCET


def main(cfg_content, ptx_content, bb_content, gpu_params_content):
    edges, nodes = parse_cfg_edges(cfg_content)

    loops = extract_loops(ptx_content)

    bb_instructions, bb_labels = map_basic_blocks(bb_content)

    loop_blocks = find_loop_blocks(ptx_content, bb_content, loops)

    dag_edges = break_loops(edges, loop_blocks)

    topo_order = topological_sort(dag_edges)

    pred_map = defaultdict(list)
    for u, v in edges:
        pred_map[v].append(u)

    block_counts = compute_block_counts(loops, loop_blocks, nodes)

    unrolled_path, path_marks = unroll_loops(topo_order, loops, loop_blocks)

    predecessors = nodepredecessor(edges, unrolled_path, path_marks, loop_blocks)

    gpu_config = GPUConfig(gpu_params_content, bb_content)

    allocator = Warp_Block_SM_Allocate(gpu_config)
    warps, sm_blocks = allocator.organize_threads_and_blocks()
    warp_allo_report = allocator.generate_assignment_report()

    warp_analyzer = WarpMemoryAnalyzer(gpu_config, unrolled_path, bb_instructions, predecessors, bb_labels)

    warpstemp = []
    num = 0

    for warp in warps:
        print(f"warps: {len(warps)}, -  warp id: {num}")
        sys.stdout.flush()

        warp_id = warp['warp_id']
        warp_threads = warp['threads']
        warpstemp.append(warp)

        warp_analyzer.set_warp_threads(warp_id, warp_threads)

        warp_analyzer.analyze_warp()

        num = num + 1

    inst_warp_report = warp_analyzer.generate_warp_report()

    syc_warp_analyzer = SycWarpWCET(gpu_config, unrolled_path, predecessors, warp_analyzer.memory_accesses, warpstemp)
    syc_warp_analyzer.analyze()
    syc_warp_report = syc_warp_analyzer.generate_report()

    multi_warp_analyse = MultiWarpWCET(gpu_config, unrolled_path, predecessors, warp_analyzer.memory_accesses, warpstemp)
    multi_warp_analyse.analyze_inter_block()
    multi_warp_report = multi_warp_analyse.generate_report()

    RTSS_warp_analyse = RTSSWarpWCET(gpu_config, unrolled_path, predecessors, warp_analyzer.memory_accesses, warpstemp)
    RTSS_warp_analyse.analyze_inter_block()
    RTSS_warp_report = RTSS_warp_analyse.generate_report()

    ZW2018_analyse = ZW2018WarpWCET(gpu_config, unrolled_path, predecessors, warp_analyzer.memory_accesses, warpstemp)
    ZW2018_report = ZW2018_analyse.generate_report()

    return block_counts, loops, loop_blocks, topo_order, unrolled_path, warp_allo_report, inst_warp_report, syc_warp_report, multi_warp_report, RTSS_warp_report, ZW2018_report

if __name__ == "__main__":
    input_dir = "input"

    with open(os.path.join(input_dir, 'CFG_graph.txt'), 'r') as f:
        cfg_content = f.read()

    with open(os.path.join(input_dir, 'ptx_annotated.txt'), 'r') as f:
        ptx_content = f.read()

    with open(os.path.join(input_dir, 'Basic_Block_instrction.txt'), 'r') as f:
        bb_content = f.read()

    with open(os.path.join(input_dir, 'gpu_parameter.txt'), 'r') as f:
        gpu_params_content = f.read()

    (block_counts, loops, loop_blocks, topo_order, unrolled_path, warp_allo_report, inst_warp_report, syc_warp_report, multi_warp_report, RTSS_warp_report, ZW2018_report) = (
        main(cfg_content, ptx_content, bb_content, gpu_params_content))

    print("Identified loops:")
    for loop_id, blocks in loop_blocks.items():
        bound = loops.get(loop_id, {}).get('bound', 'unknown')
        depth = loops.get(loop_id, {}).get('depth', 'unknown')
        print(f"Loop {loop_id} (depth={depth}, bound={bound}) contains blocks: {sorted(blocks)}")

    print("\nBlock execution counts:")
    print(block_counts)

    print("\nTopological order (min-node first):")
    print(topo_order)

    print("\nUnrolled execution path:")
    print(unrolled_path)

    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "loop.txt"), "w") as f:
        f.write(f"loop_id, depth, bound, block_number: \n")
        for loop_id, blocks in loop_blocks.items():
            bound = loops.get(loop_id, {}).get('bound', 'unknown')
            depth = loops.get(loop_id, {}).get('depth', 'unknown')
            block_count = len(blocks)
            f.write(f"{loop_id} {depth} {bound} {block_count} {sorted(blocks)}\n")

    with open(os.path.join(report_dir, 'Warp_Block_SM_Allocate.txt'), 'w') as f:
        f.write(warp_allo_report)

    with open(os.path.join(report_dir, "single_warp_analysis_summary.txt"), "w") as f:
        f.write(inst_warp_report)

    with open(os.path.join(report_dir, "syc_warp_analyzer_summary.txt"), "w") as f:
        f.write(syc_warp_report)

    with open(os.path.join(report_dir, "multi_warp_analyzer_summary.txt"), "w") as f:
        f.write(multi_warp_report)

    with open(os.path.join(report_dir, "RTSS_warp_analyzer_summary.txt"), "w") as f:
        f.write(RTSS_warp_report)

    with open(os.path.join(report_dir, "ZW2018_warp_analyzer_summary.txt"), "w") as f:
        f.write(ZW2018_report)
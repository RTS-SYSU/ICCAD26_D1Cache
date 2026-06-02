import sys
from sortedcontainers import SortedList
from collections import defaultdict, deque
import math

class MultiWarpWCET:
    def __init__(self, gpu_config, unrolled_path, predecessors, memory_accesses, warps):
        self.gpu_config = gpu_config
        self.unrolled_path = unrolled_path
        self.predecessors = predecessors
        self.thread_block_accesses = self.div_memory_accesses_block(memory_accesses)
        self.warps = warps

        self.cache_line_size = gpu_config.cache_config['line_size']
        self.cache_associativity = gpu_config.cache_config['associativity']
        self.cache_num_sets = gpu_config.cache_config['num_sets']

        self.results = []
        self.access_stats = defaultdict(lambda: {'hit': 0, 'miss': 0})

        self.cache_states = [
            [
                defaultdict(lambda: deque(maxlen=self.cache_associativity))
                for _ in range(self.cache_num_sets + 1)
            ]
            for _ in range(unrolled_path[-1] + 1)
        ]
        self.cache_stats = {
            'total_accesses': 0,
            'hits': 0,
            'misses': 0,
        }

        self.hit_addr = [SortedList(key=lambda x: x['cnt']) for _ in range(self.cache_num_sets + 1)]
        self.all_addr = [[]for _ in range(self.cache_num_sets + 1)]

    def cache_init(self):
        self.cache_states = [
            [
                defaultdict(lambda: deque(maxlen=self.cache_associativity))
                for _ in range(self.cache_num_sets + 1)
            ]
            for _ in range(self.unrolled_path[-1] + 1)
        ]

    def div_memory_accesses_block(self, memory_accesses):
        thread_block_accesses = [[[] for _ in range(len(self.unrolled_path))] for _ in range(self.gpu_config.total_blocks)]
        for pos in range(len(self.unrolled_path)):
            for idmemory in range(len(memory_accesses[pos])):
                for accessset in memory_accesses[pos][idmemory]:
                    thread_block_id = accessset[0]
                    while len(thread_block_accesses[thread_block_id][pos]) <= idmemory:
                        thread_block_accesses[thread_block_id][pos].append([])
                    thread_block_accesses[thread_block_id][pos][idmemory].append(accessset)
        return thread_block_accesses

    def merge_cache_states(self, cache_states_list):
        if not cache_states_list:
            return defaultdict(lambda: deque(maxlen=self.cache_associativity))

        if len(cache_states_list) == 1:
            return cache_states_list[0].copy()

        merged_lines = {}
        if cache_states_list:
            cache_maps = [dict(cache_state) for cache_state in cache_states_list]

            common_tags = set(cache_maps[0].keys())
            for cache_map in cache_maps[1:]:
                common_tags = common_tags.intersection(cache_map.keys())

            for tag in common_tags:
                max_age = max(cache_map[tag] for cache_map in cache_maps)
                merged_lines[tag] = max_age

        sorted_lines = sorted(merged_lines.items(), key=lambda x: x[1])
        return sorted_lines

    def compute_cache_set_and_tag(self, address):
        set_id = self.cache_num_sets

        if isinstance(address, (int, float)):
            line_size = self.cache_line_size * self.cache_num_sets
            tag = address // line_size
            set_id = address // self.cache_line_size % self.cache_num_sets
        else:
            tag = address

        return set_id, tag

    def access_cache(self, cache_state, address, access_type):
        set_id, tag = self.compute_cache_set_and_tag(address)

        hit_index = -1
        current_age = self.cache_associativity
        for i, (tag2, age) in enumerate(cache_state[set_id]):
            if tag2 == tag and age < self.cache_associativity:
                hit_index = i
                current_age = age
                break

        if hit_index >= 0:
            new_cache_list = []
            for i, (tag2, age) in enumerate(cache_state[set_id]):
                if i == hit_index:
                    new_cache_list.append((tag2, 0))
                elif age > current_age:
                    new_cache_list.append((tag2, age))
                elif age + 1 < self.cache_associativity:
                    new_cache_list.append((tag2, age + 1))
            cache_state[set_id] = new_cache_list
            result = "HIT"

        else:
            result = "MISS"
            if access_type == "LOAD":
                new_cache_list = []

                new_cache_list.append((tag, 0))

                for i, (tag2, age) in enumerate(cache_state[set_id]):
                    if age + 1 < self.cache_associativity:
                        new_cache_list.append((tag2, age + 1))
                cache_state[set_id] = new_cache_list

        return cache_state, result, current_age, set_id, tag

    def analyze_inter_block(self):
        sys.stdout.flush()
        for thread_block_id in range(len(self.thread_block_accesses)):
            print(f"intra-block analyze: {len(self.thread_block_accesses)}, : {thread_block_id}")
            sys.stdout.flush()
            self.analyze_intra_block(self.thread_block_accesses[thread_block_id], thread_block_id)
            self.cache_init()

        for set_id in range(self.cache_num_sets):

            use_addr_dict = {}
            for addr_info in self.hit_addr[set_id]:
                key = addr_info['id']
                use_addr_dict[key] = set()

            for i in range(len(self.all_addr[set_id])):

                print(f"set_id and intra-analyze: {set_id}, : {len(self.all_addr[set_id])}, : {i}")
                sys.stdout.flush()

                tag = self.all_addr[set_id][i]['tag']
                thread_block_id = self.all_addr[set_id][i]['thread_block_id']

                use_addr = set()

                to_modify = []
                for j, addr_info in enumerate(self.hit_addr[set_id]):
                    hit_tag = addr_info['tag']
                    hit_thread_block_id = addr_info['thread_block_id']
                    key = addr_info['id']

                    if  tag in use_addr_dict[key]:
                        continue

                    if hit_thread_block_id != thread_block_id and hit_tag != tag and hit_tag not in use_addr:
                        use_addr.add(hit_tag)
                        use_addr_dict[key].add(tag)
                        to_modify.append((j, addr_info))
                        if len(use_addr) >= self.cache_associativity:
                            break

                for j, addr_info in reversed(to_modify):
                    self.hit_addr[set_id].remove(addr_info)

                for j, addr_info in reversed(to_modify):
                    if addr_info['cnt'] > 0:
                        addr_info['cnt'] -= 1
                        self.hit_addr[set_id].add(addr_info)

        for set_id in range(self.cache_num_sets+1):
            self.cache_stats['hits'] += len(self.hit_addr[set_id])

        self.cache_stats['misses'] = self.cache_stats['total_accesses'] - self.cache_stats['hits']

    def analyze_intra_block(self, memory_accesses, thread_block_id):
        total_hit_num = 0
        for pos in range(len(self.unrolled_path)):
            basic_block_id = self.unrolled_path[pos]

            preds = self.predecessors[pos]

            if preds:
                for set_id in range(self.cache_num_sets + 1):
                    pred_cache_states = [self.cache_states[p][set_id] for p in preds]
                    self.cache_states[basic_block_id][set_id] = self.merge_cache_states(pred_cache_states)

            for idmemory in range(len(memory_accesses[pos])):

                print(f"intra block {thread_block_id}, {len(self.unrolled_path)}, {len(memory_accesses[pos])} : {pos}, {idmemory}")
                sys.stdout.flush()

                address_dict = {}
                access_type = 'LOAD'
                for accessset in memory_accesses[pos][idmemory]:
                    access_type = accessset[1]
                    addrset = accessset[2]
                    for addr in addrset:
                        set_id, tag = self.compute_cache_set_and_tag(addr)
                        address_dict[(set_id, tag)] = address_dict.get((set_id, tag), 0) + 1

                current_cache_state = self.cache_states[basic_block_id].copy()

                address_list = [(set_id, tag, cnt) for (set_id, tag), cnt in address_dict.items()]
                num_set_in_list = [0 for i in range(self.cache_num_sets)]

                for address in address_list:
                    (set_id, tag, cnt) = address
                    num_set_in_list[set_id] = num_set_in_list[set_id] + 1
                    address_list2 = []
                    for address2 in address_list:
                        (set_id2, tag2, cnt2) = address2
                        if set_id2 == set_id and tag2 != tag:
                            address_list2.append((set_id2, tag2, cnt2))

                    self.cache_stats['total_accesses'] += cnt

                    for i in range(cnt):
                        addr_info = {
                            'tag': tag,
                            'cnt': 0,
                            'thread_block_id': thread_block_id,
                        }
                        self.all_addr[set_id].append(addr_info)

                    addr_age = self.cache_associativity
                    for i, (tag2, age) in enumerate(current_cache_state[set_id]):
                        if tag2 == tag:
                            addr_age = age
                            break

                    cur_addr = (tag * self.cache_num_sets + set_id) * self.cache_line_size

                    if addr_age + len(address_list2)  >= self.cache_associativity:
                        if access_type != "LOAD":
                            self.cache_stats['misses'] += cnt
                            cnt = 0
                        else:
                            address_list2 = [(set_id, tag, cnt) for set_id, tag, cnt in address_list2 if cnt > 0]
                            self.cache_stats['misses'] += 1
                            cnt = cnt - 1

                            while cnt > 0 and len(address_list2) >= self.cache_associativity:
                                self.cache_stats['misses'] += 1
                                cnt = cnt - 1
                                sorted_list = sorted(address_list2, key=lambda x: x[2], reverse=True)

                                k = self.cache_associativity
                                for i in range(min(k, len(sorted_list))):
                                    set_id, tag, cnt = sorted_list[i]
                                    sorted_list[i] = (set_id, tag, cnt - 1)

                                address_list2 = [(set_id, tag, cnt) for set_id, tag, cnt in sorted_list if cnt > 0]

                    addr_info = {
                        'tag': tag,
                        'cnt': self.cache_associativity - len(address_list2),
                        'thread_block_id': thread_block_id,
                        'id': total_hit_num
                    }
                    if cnt > 0:
                        self.hit_addr[set_id].add(addr_info)
                        cnt -= 1
                        total_hit_num += 1

                    while cnt > 0:
                        addr_info = {
                            'tag': tag,
                            'cnt': self.cache_associativity,
                            'thread_block_id': thread_block_id,
                            'id': total_hit_num
                        }
                        self.hit_addr[set_id].add(addr_info)
                        cnt = cnt - 1
                        total_hit_num += 1

                    new_cache_state, cache_result, back_age, set_id, tag = self.access_cache(current_cache_state,
                                                                                             cur_addr, access_type)
                    current_cache_state = new_cache_state

                for address in address_list:
                    set_id, tag, cnt = address
                    for i, (tag2, age) in enumerate(current_cache_state[set_id]):
                        if tag2 == tag:
                            current_cache_state[set_id][i] = (tag2, num_set_in_list[set_id] - 1)
                            break
                self.cache_states[basic_block_id] = current_cache_state

    def generate_report(self):
        report = []
        report.append("Multi-Warp WCET Analysis Report")
        total_access = max(1,self.cache_stats['total_accesses'])
        total_miss = self.cache_stats['misses']
        total_hit = self.cache_stats['hits']

        report.append(f"Warp Number: {len(self.warps)}, block size: {self.gpu_config.block_dim[0] * self.gpu_config.block_dim[1] * self.gpu_config.block_dim[2]}")
        report.append(f"Total Accesses: {total_access}")
        report.append(f"Multi warp Hits: {total_hit} ({total_hit / total_access * 100:.2f}%)")
        report.append(f"Multi warp Misses: {total_miss} ({total_miss / total_access * 100:.2f}%)")

        return "\n".join(report)
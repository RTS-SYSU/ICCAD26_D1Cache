class Warp_Block_SM_Allocate:
    def __init__(self, gpu_config):
        self.gpu_config = gpu_config
        self.warps = []
        self.sm_blocks = {}
        self.thread_mapping = {}

        self.threads_per_block = gpu_config.threads_per_block
        self.warps_per_block = gpu_config.warps_per_block
        self.total_blocks = gpu_config.total_blocks

        self.sm_blocks = {sm_id: [] for sm_id in range(gpu_config.num_sm)}

    def organize_threads_and_blocks(self):
        block_count = 0
        warp_count = 0
        thread_in_warp = 0

        current_block_threads = []
        current_warp_threads = []

        for bz in range(self.gpu_config.grid_dim[2]):
            for by in range(self.gpu_config.grid_dim[1]):
                for bx in range(self.gpu_config.grid_dim[0]):
                    block_id = (bx, by, bz)
                    sm_id = block_count % self.gpu_config.num_sm
                    self.sm_blocks[sm_id].append(block_id)

                    thread_in_block = 0
                    warp_in_block = 0

                    for tz in range(self.gpu_config.block_dim[2]):
                        for ty in range(self.gpu_config.block_dim[1]):
                            for tx in range(self.gpu_config.block_dim[0]):
                                thread_id = (bx, by, bz, tx, ty, tz)
                                current_block_threads.append(thread_id)

                                current_warp_threads.append(thread_id)
                                thread_in_warp += 1

                                if thread_in_warp == self.gpu_config.warp_size:
                                    warp_id = (bx, by, bz, warp_in_block)
                                    self.warps.append({
                                        'warp_id': warp_id,
                                        'block_id': block_id,
                                        'threads': current_warp_threads,
                                        'sm_id': sm_id
                                    })

                                    current_warp_threads = []
                                    thread_in_warp = 0
                                    warp_in_block += 1
                                    warp_count += 1

                                thread_in_block += 1

                    if current_warp_threads:
                        warp_id = (bx, by, bz, warp_in_block)
                        self.warps.append({
                            'warp_id': warp_id,
                            'block_id': block_id,
                            'threads': current_warp_threads,
                            'sm_id': sm_id
                        })
                        warp_in_block += 1
                        warp_count += 1
                        current_warp_threads = []
                        thread_in_warp = 0

                    block_count += 1

        return self.warps, self.sm_blocks

    def get_thread_id_str(self, thread_id):
        bx, by, bz, tx, ty, tz = thread_id
        return f"({tx},{ty},{tz})"

    def get_max_active_blocks(self):
        regs_per_thread = len(self.gpu_config.total_registers)
        threads_per_block = self.gpu_config.threads_per_block
        regs_per_block = regs_per_thread * threads_per_block
        max_blocks_by_regs = self.gpu_config.max_registers_per_sm // regs_per_block

        warps_per_block = (threads_per_block + self.gpu_config.warp_size - 1) // self.gpu_config.warp_size
        max_blocks_by_warps = self.gpu_config.max_warps_per_sm // warps_per_block

        max_active_blocks = min(max_blocks_by_regs, max_blocks_by_warps)

        return max_active_blocks

    def generate_assignment_report(self):
        report = []

        report.append("GPU Configuration:")
        report.append(f"  Grid Dimensions: {self.gpu_config.grid_dim}")
        report.append(f"  Block Dimensions: {self.gpu_config.block_dim}")
        report.append(f"  Number of SMs: {self.gpu_config.num_sm}")
        report.append(f"  Warp Size: {self.gpu_config.warp_size}")

        report.append("\nThread Block to SM Assignment:")
        for sm_id, blocks in self.sm_blocks.items():
            report.append(f"  SM {sm_id}: {len(blocks)} blocks")
            if blocks:
                if len(blocks) <= 4:
                    block_ids = ", ".join(str(b) for b in blocks)
                else:
                    block_ids = f"{blocks[0]}, {blocks[1]}, ..., {blocks[-2]}, {blocks[-1]}"
                report.append(f"    Block IDs: {block_ids}")

        report.append("\nWarp Information: ( id = (bx, by, bz, warp_in_block) )")
        for warp in self.warps:
            first_thread = warp['threads'][0]
            last_thread = warp['threads'][-1]

            first_thread_str = self.get_thread_id_str(first_thread)
            last_thread_str = self.get_thread_id_str(last_thread)

            report.append(f"  Warp {warp['warp_id']} in Block {warp['block_id']} assigned to SM {warp['sm_id']}")
            report.append(f"    Threads: {first_thread_str} to {last_thread_str} ({len(warp['threads'])} threads)")

        return "\n".join(report)

    def get_special_reg_values(self, thread_id):
        bx, by, bz, tx, ty, tz = thread_id

        return {
            'tid_x': tx,
            'tid_y': ty,
            'tid_z': tz,
            'ctaid_x': bx,
            'ctaid_y': by,
            'ctaid_z': bz,
        }
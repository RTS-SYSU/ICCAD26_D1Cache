import re

class GPUConfig:
    def __init__(self, params_content, bb_content):
        self.grid_dim = [1, 1, 1]
        self.block_dim = [1, 1, 1]
        self.cache_config = {}
        self.num_sm = 1
        self.max_registers_per_sm = 65536
        self.max_warps_per_sm = 64
        self.warp_size = 32
        self.val_values = {}
        self.bb_content = bb_content
        self.register_usage = {}
        self.total_registers = 0

        self._parse_params(params_content)
        self._parse_params_val(params_content)
        self._analyze_ptx_registers()

    def _parse_params(self, params_content):
        self.cache_config.setdefault('associativity', 256)
        self.cache_config.setdefault('num_sets', 4)
        self.cache_config.setdefault('line_size', 128)

        for line in params_content.split('\n'):
            if line.startswith('-gridDim.x'):
                self.grid_dim[0] = int(line.split()[-1])
            elif line.startswith('-gridDim.y'):
                self.grid_dim[1] = int(line.split()[-1])
            elif line.startswith('-gridDim.z'):
                self.grid_dim[2] = int(line.split()[-1])
            elif line.startswith('-blockDim.x'):
                self.block_dim[0] = int(line.split()[-1])
            elif line.startswith('-blockDim.y'):
                self.block_dim[1] = int(line.split()[-1])
            elif line.startswith('-blockDim.z'):
                self.block_dim[2] = int(line.split()[-1])
            elif line.startswith('-gpgpu_cache:dl1'):
                config_str = line.split("dl1", 1)[-1].strip()
                params = config_str.split(':')
                if len(params) >= 4:
                    self.cache_config['num_sets'] = int(params[1])
                    self.cache_config['line_size'] = int(params[2])
                    self.cache_config['associativity'] = int(params[3])
            elif line.startswith('-gpgpu_n_clusters'):
                self.num_sm = int(line.split()[-1])
            elif line.startswith('-gpgpu_shader_registers'):
                self.max_registers_per_sm = int(line.split()[-1])
            elif line.startswith('-gpgpu_warp_clusters'):
                self.max_warps_per_sm = int(line.split()[-1])

    def _parse_params_val(self, params_content):
        lines = params_content.strip().split('\n')
        in_val_section = False
        for line in lines:
            line = line.strip()
            if line == 'Val:':
                in_val_section = True
                continue
            if in_val_section:
                if not line or not line.startswith('-'):
                    continue
                parts = line[1:].split()
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    self.val_values[key] = value

    def _analyze_ptx_registers(self):
        register_patterns = {
            'r': r'%r(\d+)',
            'rd': r'%rd(\d+)',
            'f': r'%f(\d+)',
            'p': r'%p(\d+)',
        }
        for reg_type in register_patterns.keys():
            self.register_usage[reg_type] = set()
        lines = self.bb_content.strip().split('\n')

        for line in lines:
            for reg_type, pattern in register_patterns.items():
                matches = re.findall(pattern, line)
                for match in matches:
                    self.register_usage[reg_type].add(int(match))

        self.total_registers = sum(len(reg_set) for reg_set in self.register_usage.values())

    @property
    def threads_per_block(self):
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]

    @property
    def warps_per_block(self):
        return (self.threads_per_block + self.warp_size - 1) // self.warp_size

    @property
    def total_blocks(self):
        return self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]
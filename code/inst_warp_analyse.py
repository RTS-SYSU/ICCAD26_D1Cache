import math
import re
import struct

class WarpMemoryAnalyzer:
    def __init__(self, gpu_config, unrolled_path, bb_instructions, predecessors, bb_labels):
        self.gpu_config = gpu_config
        self.unrolled_path = unrolled_path
        self.bb_instructions = bb_instructions
        self.predecessors = predecessors
        self.unrolled_path = unrolled_path
        self.bb_labels = bb_labels
        self.begAdrr = 3221225472

        self.register_states = [[{} for _ in range(32)] for _ in range(len(bb_instructions) + 1)]
        self.memory_accesses = [[] for _ in range(len(unrolled_path))]

        self.unknown_counter = len(self.gpu_config.val_values) - 1

        self.para_mater = self.gpu_config.val_values
        self.init_para_mater()

        self.current_warp_threads = []
        self.current_warp_id = (0,0,0)

        self.cache_line_size = gpu_config.cache_config['line_size']
        self.cache_associativity = gpu_config.cache_config['associativity']
        self.cache_num_sets = gpu_config.cache_config['num_sets']


    def init_para_mater(self):
        for param_name in self.para_mater:
            value = self.para_mater[param_name]
            if value.startswith('uu'):
                self.para_mater[param_name] = self.begAdrr
                self.begAdrr += 3221225472

    def set_warp_threads(self, warp_id, warp_threads):
        self.register_states = [[{} for _ in range(32)] for _ in range(len(self.bb_instructions) + 1)]
        self.current_warp_threads = warp_threads
        self.current_warp_id = warp_id
        for idx, thread_id in enumerate(warp_threads):
            bx, by, bz, tx, ty, tz = thread_id
            self.register_states[0][idx] = {}
            self.register_states[0][idx]['%tid.x'] = tx
            self.register_states[0][idx]['%tid.y'] = ty
            self.register_states[0][idx]['%tid.z'] = tz
            self.register_states[0][idx]['%ctaid.x'] = bx
            self.register_states[0][idx]['%ctaid.y'] = by
            self.register_states[0][idx]['%ctaid.z'] = bz
            self.register_states[0][idx]['%ntid.x'] = self.gpu_config.block_dim[0]
            self.register_states[0][idx]['%ntid.y'] = self.gpu_config.block_dim[1]
            self.register_states[0][idx]['%ntid.z'] = self.gpu_config.block_dim[2]
            self.register_states[0][idx]['%nctaid.x'] = self.gpu_config.grid_dim[0]
            self.register_states[0][idx]['%nctaid.y'] = self.gpu_config.grid_dim[1]
            self.register_states[0][idx]['%nctaid.z'] = self.gpu_config.grid_dim[2]

    def get_register_value(self, idx, pos, reg_name):
        if reg_name in self.register_states[pos][idx]:
            return self.register_states[pos][idx][reg_name]

        self.unknown_counter += 1
        self.register_states[pos][idx][reg_name] = f"u_{self.unknown_counter}"
        return self.register_states[pos][idx][reg_name]

    def set_register_value(self, pos, idx, reg_name, value):
        self.register_states[pos][idx][reg_name] = value

    def merge_register_states(self, states):
        merged = {}

        for state in states:
            for reg, value in state.items():
                if reg in merged:
                    if merged[reg] != value:
                        self.unknown_counter += 1
                        merged[reg] = f"u_{self.unknown_counter}"
                else:
                    merged[reg] = value

        return merged

    def evaluate_expression(self, expr, thread_id, pos):
        if isinstance(expr, (int, float)):
            return expr

        if isinstance(expr, str) and expr.startswith('0f'):
            try:
                hex_str = expr[2:].strip()
                if len(hex_str) == 8:
                    return struct.unpack('>f', bytes.fromhex(hex_str))[0]
                elif len(hex_str) == 16:
                    return struct.unpack('>d', bytes.fromhex(hex_str))[0]
            except:
                pass

        if isinstance(expr, str) and expr.startswith('u_'):
            return expr

        pattern = r'%[a-zA-Z0-9_.]+'
        registers = re.findall(pattern, expr)

        evaluated_expr = expr
        for reg in registers:
            reg_value = self.get_register_value(thread_id, pos, reg)
            evaluated_expr = evaluated_expr.replace(reg, str(reg_value))

        try:
            if 'u_' in evaluated_expr:
                return evaluated_expr
            return eval(evaluated_expr)
        except:
            return evaluated_expr

    def create_expression(self, op, *args):
        has_unknown = any(isinstance(arg, str) and 'u_' in arg for arg in args)

        if has_unknown:
            if op == 'add':
                if args[1] == 0:
                    return args[0]
                return f"({args[0]} + {args[1]})"
            elif op == 'sub':
                return f"({args[0]} - {args[1]})"
            elif op == 'mul':
                if args[1] == 1:
                    return args[0]
                return f"({args[0]} * {args[1]})"
            elif op == 'not':
                return f"(!{args[0]})"
            elif op == 'and':
                return f"({args[0]} & {args[1]})"
            elif op == 'or':
                return f"({args[0]} | {args[1]})"
            elif op == 'xor':
                return f"({args[0]} ^ {args[1]})"
            elif op == 'shl':
                return f"({args[0]} << {args[1]})"
            elif op == 'shr':
                return f"({args[0]} >> {args[1]})"
            elif op == 'mad':
                return f"({args[0]} * {args[1]} + {args[2]})"
            elif op == 'div':
                return f"({args[0]} / {args[1]})"
            elif op == 'ge':
                return f"({args[0]} >= {args[1]})"
            elif op == 'le':
                return f"({args[0]} <= {args[1]})"
            elif op == 'lt':
                return f"({args[0]} < {args[1]})"
            elif op == 'gt':
                return f"({args[0]} > {args[1]})"
            elif op == 'eq':
                return f"({args[0]} == {args[1]})"
            elif op == 'ne':
                return f"({args[0]} != {args[1]})"
            elif op == 'min':
                return f"min({args[0]},{args[1]})"
            elif op == 'max':
                return f"max({args[0]},{args[1]})"
            elif op == 'rem':
                return f"({args[0]} % {args[1]})"
            else:
                return f"{op}({', '.join(map(str, args))})"
        else:
            try:
                if op == 'add':
                    return args[0] + args[1]
                elif op == 'sub':
                    return args[0] - args[1]
                elif op == 'mul':
                    return args[0] * args[1]
                elif op == 'and':
                    return args[0] & args[1]
                elif op == 'or':
                    return args[0] | args[1]
                elif op == 'not':
                    return not args[0]
                elif op == 'xor':
                    return args[0] ^ args[1]
                elif op == 'shl':
                    return args[0] << args[1]
                elif op == 'shr':
                    return args[0] >> args[1]
                elif op == 'mad':
                    return args[0] * args[1] + args[2]
                elif op == 'div':
                    if args[1] == 0:
                        return "division_by_zero"
                    return args[0] / args[1]
                elif op == 'ge':
                    return args[0] >= args[1]
                elif op == 'le':
                    return args[0] <= args[1]
                elif op == 'lt':
                    return args[0] < args[1]
                elif op == 'gt':
                    return args[0] > args[1]
                elif op == 'eq':
                    return args[0] == args[1]
                elif op == 'ne':
                    return args[0] != args[1]
                elif op == 'min':
                    return min(args[0], args[1])
                elif op == 'max':
                    return max(args[0], args[1])
                elif op == 'rem':
                    return args[0] % args[1]
                else:
                    return f"{op}({', '.join(map(str, args))})"
            except Exception as e:
                return f"{op}({', '.join(map(str, args))})"

    def process_instruction(self, instr, idx, thread_id, pos, instr_index, instr_text):
        parts = re.split(r'[\s,]+', instr)
        if not parts:
            return False

        opcode = parts[0]
        operands = [p.rstrip(';') for p in parts[1:] if p]

        if opcode.startswith('@') or opcode == 'bra':
            return self.record_branch_instruction(opcode, operands, idx, pos)

        try:
            if opcode.startswith('ld.param'):
                self.process_ld_param(operands, idx, pos)

            elif opcode.startswith('mov'):
                self.process_mov_op(operands, idx, pos)

            elif opcode.startswith(('add', 'sub', 'mul', 'and', 'xor', 'min', 'max', 'or', 'not', 'rem')):
                self.process_alu_op(operands, idx, pos, opcode)

            elif opcode.startswith('mad'):
                self.process_mad_op(operands, idx, pos)

            elif opcode.startswith(('shl', 'shr')):
                self.process_shift_op(operands, idx, pos, opcode)

            elif opcode.startswith('cvta.to.global'):
                self.process_cvta_op(operands, idx, pos)

            elif opcode.startswith('ld.global'):
                return self.process_memory_access(opcode, operands, idx, thread_id, pos, instr_index, instr_text)

            elif opcode.startswith('st.global'):
                return self.process_memory_access(opcode, operands, idx, thread_id, pos, instr_index, instr_text)

            elif opcode.startswith('setp'):
                self.process_setp_op(operands, idx, pos, opcode)

            elif opcode.startswith('div'):
                self.process_div_op(operands, idx, pos, opcode)

            elif opcode.startswith('bar.sync'):
                pass

            elif opcode.startswith('neg'):
                self.process_neg_op(operands, idx, pos)

            elif opcode.startswith('cvt.rn.f32.s32'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_cvtrn_op(operands, idx, pos, 0)

            elif opcode.startswith('cvt.rn.s32.f32'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_cvtrn_op(operands, idx, pos, 1)

            elif opcode.startswith('cvt.rzi.f32.s32'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_cvtrzi_op(operands, idx, pos, 0)

            elif opcode.startswith('cvt.rzi.s32.f32'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_cvtrzi_op(operands, idx, pos, 1)

            elif opcode.startswith('fma'):
                self.process_fma_op(operands, idx, pos)

            elif opcode.startswith('cvt.s64.s32'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_cvt_op(operands, idx, pos)

            elif opcode.startswith('selp'):
                tempinstr = instr.replace('%', ' %')
                tempparts = re.split(r'[\s,]+', tempinstr)
                operands = [p.rstrip(';') for p in tempparts[1:] if p]
                self.process_selp_op(operands, idx, pos)

        except Exception as e:
            pass

        return False

    def record_branch_instruction(self, opcode, operands, idx, pos):
        target = operands[-1] if operands else "unknown"

        predicate = None
        predicate_value = None
        invert = False
        if opcode.startswith('@'):
            predicate_match = re.match(r'@(!?)(%p\d+)', opcode)
            if predicate_match:
                invert = (predicate_match.group(1) == '!')
                predicate = predicate_match.group(2)
                predicate_value = self.get_register_value(idx, pos, predicate)
                if invert and isinstance(predicate_value, (bool, int)):
                    predicate_value = not predicate_value
            opcode = 'bra'

        jump_taken = False
        if predicate_value is not None:
            if isinstance(predicate_value, (int, bool)):
                jump_taken = ('@', self.bb_labels[target]) if predicate_value else ('@', pos + 1)
            elif isinstance(predicate_value, str) and predicate_value.startswith('u_'):
                jump_taken = False

        return jump_taken

    def process_cvta_op(self, operands, idx, pos):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        src_reg = operands[1]

        src_value = self.get_register_value(idx, pos, src_reg)

        self.set_register_value(pos, idx, dest_reg, src_value)

    def process_cvtrn_op(self, operands, idx, pos, type):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        src_reg = operands[1]

        src_value = self.get_register_value(idx, pos, src_reg)

        if type == 1:
            src_value = round(src_value)
        else:
            src_value = float(src_value)

        self.set_register_value(pos, idx, dest_reg, src_value)

    def process_cvtrzi_op(self, operands, idx, pos, type):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        src_reg = operands[1]

        src_value = self.get_register_value(idx, pos, src_reg)

        if type == 1:
            src_value = math.trunc(src_value)
        else:
            src_value = float(src_value)

        self.set_register_value(pos, idx, dest_reg, src_value)

    def process_cvt_op(self, operands, idx, pos):
        self.process_mov_op(operands, idx, pos)

    def process_setp_op(self, operands, idx, pos, opcode):
        percent_index = opcode.find('%')
        if percent_index > 0:
            reg_part = [opcode[percent_index:]]
            operands = reg_part + operands

        if len(operands) < 3:
            return

        pred_reg = operands[0]
        src1 = self.evaluate_expression(operands[1], idx, pos)
        src2 = self.evaluate_expression(operands[2], idx, pos)

        if '.le.' in opcode:
            comp_type = 'le'
        elif '.lt.' in opcode:
            comp_type = 'lt'
        elif '.gt.' in opcode:
            comp_type = 'gt'
        elif '.ge.' in opcode:
            comp_type = 'ge'
        elif '.eq.' in opcode:
            comp_type = 'eq'
        elif '.ne.' in opcode:
            comp_type = 'ne'
        else:
            comp_type = 'eq'

        result = self.create_expression(comp_type, src1, src2)
        self.set_register_value(pos, idx, pred_reg, result)

    def process_selp_op(self, operands, idx, pos):
        if len(operands) < 4:
            return

        dest_reg = operands[0]
        src_a = self.evaluate_expression(operands[1], idx, pos)
        src_b = self.evaluate_expression(operands[2], idx, pos)
        pred_reg = operands[3]

        pred_value = self.get_register_value(idx, pos, pred_reg)

        if isinstance(pred_value, (bool, int)):
            result = src_a if pred_value else src_b
        else:
            result = f"({src_a} if {pred_value} else {src_b})"

        self.set_register_value(pos, idx, dest_reg, result)

    def process_div_op(self, operands, idx, pos, opcode):
        if len(operands) < 3:
            return

        dest_reg = operands[0]
        src1 = self.evaluate_expression(operands[1], idx, pos)
        src2 = self.evaluate_expression(operands[2], idx, pos)

        is_int_div = '.s' in opcode or '.u' in opcode

        has_unknown = any(isinstance(arg, str) and 'u_' in arg for arg in (src1, src2))

        if has_unknown:
            op_name = 'div.s' if is_int_div else 'div.f'
            result = self.create_expression(op_name, src1, src2)
        else:
            try:
                if is_int_div:
                    if src2 == 0:
                        result = "division_by_zero"
                    else:
                        if src1 * src2 >= 0:
                            result = src1 // src2
                        else:
                            result = -(-src1 // src2)
                else:
                    if src2 == 0:
                        result = "division_by_zero"
                    else:
                        result = src1 / src2
            except Exception as e:
                result = f"div({src1},{src2})"

        self.set_register_value(pos, idx, dest_reg, result)

    def process_ld_param(self, operands, idx, pos):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        param_name = operands[1].strip('[]')

        if param_name in self.para_mater:
            value = self.para_mater[param_name]
        else:
            self.unknown_counter += 1
            value = f"u_{self.unknown_counter}"

        self.set_register_value(pos, idx, dest_reg, value)

    def process_alu_op(self, operands, idx, pos, opcode):
        if len(operands) == 2:
            dest_reg = operands[0]
            src = self.evaluate_expression(operands[1], idx, pos)
            if opcode.startswith('not'):
                result = self.create_expression('not', src)
            else:
                result = f"{opcode}({src})"
            self.set_register_value(pos, idx, dest_reg, result)
            return

        if len(operands) < 3:
            return

        dest_reg = operands[0]
        src1 = self.evaluate_expression(operands[1], idx, pos)
        src2 = self.evaluate_expression(operands[2], idx, pos)

        if opcode.startswith('add'):
            result = self.create_expression('add', src1, src2)
        elif opcode.startswith('sub'):
            result = self.create_expression('sub', src1, src2)
        elif opcode.startswith('mul'):
            result = self.create_expression('mul', src1, src2)
        elif opcode.startswith('and'):
            result = self.create_expression('and', src1, src2)
        elif opcode.startswith('or'):
            result = self.create_expression('or', src1, src2)
        elif opcode.startswith('xor'):
            result = self.create_expression('xor', src1, src2)
        elif opcode.startswith('min'):
            result = self.create_expression('min', src1, src2)
        elif opcode.startswith('max'):
            result = self.create_expression('max', src1, src2)
        elif opcode.startswith('rem'):
            result = self.create_expression('rem', src1, src2)
        else:
            result = f"{src1} {opcode[:3]} {src2}"

        self.set_register_value(pos, idx, dest_reg, result)

    def process_mad_op(self, operands, idx, pos):
        if len(operands) < 4:
            return

        dest_reg = operands[0]
        src1 = self.evaluate_expression(operands[1], idx, pos)
        src2 = self.evaluate_expression(operands[2], idx, pos)
        src3 = self.evaluate_expression(operands[3], idx, pos)

        result = self.create_expression('mad', src1, src2, src3)
        self.set_register_value(pos, idx, dest_reg, result)

    def process_fma_op(self, operands, idx, pos):
        if len(operands) < 4:
            return

        dest_reg = operands[0]
        src1 = self.evaluate_expression(operands[1], idx, pos)
        src2 = self.evaluate_expression(operands[2], idx, pos)
        src3 = self.evaluate_expression(operands[3], idx, pos)

        result = self.create_expression('mad', src1, src2, src3)
        self.set_register_value(pos, idx, dest_reg, result)

    def process_mov_op(self, operands, thread_id, pos):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        src = self.evaluate_expression(operands[1], thread_id, pos)

        self.set_register_value(pos, thread_id, dest_reg, src)

    def process_neg_op(self, operands, thread_id, pos):
        if len(operands) < 2:
            return

        dest_reg = operands[0]
        src = self.evaluate_expression(operands[1], thread_id, pos)

        self.set_register_value(pos, thread_id, dest_reg, -src)

    def process_shift_op(self, operands, idx, pos, opcode):
        if len(operands) < 3:
            return

        dest_reg = operands[0]
        src = self.evaluate_expression(operands[1], idx, pos)
        shift = self.evaluate_expression(operands[2], idx, pos)

        if opcode.startswith('shl'):
            result = self.create_expression('shl', src, shift)
        elif opcode.startswith('shr'):
            result = self.create_expression('shr', src, shift)
        else:
            result = f"{src} {opcode} {shift}"

        self.set_register_value(pos, idx, dest_reg, result)

    def process_memory_access(self, opcode, operands, idx, thread_id, pos, instr_index, instr_text):
        if len(operands) < 2:
            return False

        if opcode.startswith('ld'):
            access_type = 'LOAD'
            addr_expr = operands[1].strip('[]')
        else:
            access_type = 'STORE'
            addr_expr = operands[0].strip('[]')

        if '+' in addr_expr:
            parts = addr_expr.split('+')
            base = parts[0].strip()
            offset = parts[1].strip()
        elif '-' in addr_expr:
            parts = addr_expr.split('-')
            base = parts[0].strip()
            offset = '-' + parts[1].strip()
        else:
            base = addr_expr
            offset = '0'

        base_value = self.evaluate_expression(base, idx, pos)
        offset_value = self.evaluate_expression(offset, idx, pos)

        address = self.create_expression('add', base_value, offset_value)

        if opcode.startswith('ld'):
            dest_reg = operands[0]
            self.set_register_value(pos, idx, dest_reg, address)

        return ('addr', address, access_type)


    def analyze_warp(self):
        skip_until_label = [[] for _ in range(32)]
        for idx, thread_id in enumerate(self.current_warp_threads):
            skip_until_label[idx] = None

        for pos, block_id in enumerate(self.unrolled_path):

            preds = self.predecessors[pos]
            preds = []
            if pos > 0:
                preds.append(self.unrolled_path[pos-1])

            if preds:
                for idx, thread_id in enumerate(self.current_warp_threads):
                    pred_states = [self.register_states[p][idx] for p in preds]
                    if pred_states:
                        self.register_states[block_id][idx] = self.merge_register_states(pred_states)

            instructions = self.bb_instructions.get(block_id, [])
            idmemory = 0

            for instr_idx, instr in enumerate(instructions):
                if instr.startswith('BB'):
                    continue

                address_set = set()
                memory_type = 'LOAD'

                for idx, thread_id in enumerate(self.current_warp_threads):
                    should_skip = self.process_instruction(
                        instr, idx, thread_id, block_id, instr_idx, instr
                    )

                    if should_skip == False:
                        continue
                    elif should_skip[0] == '@':
                        skip_until_label[idx] = should_skip[1]
                    elif should_skip[0] == 'addr':
                        address_set.add(should_skip[1]//self.cache_line_size*self.cache_line_size)


                if len(address_set) > 0:
                    bx, by, bz, tx, ty, tz = self.current_warp_threads[-1]
                    block_id_int = (bx + by*self.gpu_config.grid_dim[0] +
                                    bz*(self.gpu_config.grid_dim[0]+self.gpu_config.grid_dim[1]))
                    memory_access_tuple = (block_id_int, memory_type, address_set)

                    if len(self.memory_accesses[pos]) <= idmemory:
                        temp_memory_access = []
                        temp_memory_access.append(memory_access_tuple)
                        self.memory_accesses[pos].append(temp_memory_access)
                    else:
                        self.memory_accesses[pos][idmemory].append(memory_access_tuple)

                    idmemory = idmemory + 1


    def generate_warp_report(self):
        report = []

        report.append("===Inst Warp Memory Access Analysis Report ===")

        report.append("\nGPU Configuration:")
        report.append(f"  Grid Dimensions: {self.gpu_config.grid_dim}")
        report.append(f"  Block Dimensions: {self.gpu_config.block_dim}")

        report.append("\nMemory Accesses:")
        report.append(f"    {'Address':<15}  {'set':<10}  {'tag':<10}  {'warp_id':<10} {'memory_type':<10}")

        num_set = [0,0,0,0,0]

        for i in range(len(self.memory_accesses)):
            for j in range(len(self.memory_accesses[i])):
                report.append(f"\n    pos: {i:<10} id: {j:<10}")
                memory_access_tuple = self.memory_accesses[i][j]
                for warp_id, memory_type, address_set in memory_access_tuple:
                    for address in address_set:
                        set_id, tag = self.compute_cache_set_and_tag(address)
                        report.append(f"    {address:<15}  {set_id:<10}  {tag*128*4:<10} {str(warp_id):<15} {memory_type:<10}")
                        num_set[set_id] = num_set[set_id] + 1

        for i in range(5):
            report.append(f"   num_set[{i}]: {num_set[i]:<15}")

        return "\n".join(report)

    def compute_cache_set_and_tag(self, address):
        set_id = self.cache_num_sets

        if isinstance(address, (int, float)):
            line_size = self.cache_line_size * self.cache_num_sets
            tag = address // line_size
            set_id = address // self.cache_line_size % self.cache_num_sets
        else:
            tag = address

        return set_id, tag
import sys
import math

class Cache:
    def __init__(self, name, num_ways, total_size, line_size):
        self.name = name
        self.num_ways = num_ways
        self.total_size = total_size
        self.line_size = line_size
        
        self.num_lines = total_size // line_size
        self.num_sets = self.num_lines // num_ways
        
        self.offset_bits = int(math.log2(line_size))
        self.index_bits = int(math.log2(self.num_sets))
        self.tag_bits = 32 - self.offset_bits - self.index_bits
        
        # Each set is a list of tags in LRU order (oldest at index 0).
        self.sets = [[] for _ in range(self.num_sets)]

    def parse_address(self, address_int):
        block_address = address_int >> self.offset_bits
        set_index = block_address & (self.num_sets - 1)
        tag = block_address >> self.index_bits
        return set_index, tag

    def touch_lru(self, cache_set, tag):
        for i, t in enumerate(cache_set):
            if t == tag:
                cache_set.pop(i)
                cache_set.append(tag)

    def access(self, address_int):
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        if tag in cache_set:
            self.touch_lru(cache_set, tag)
            return True
        return False

    def insert(self, address_int):
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        evicted_address = None

        if tag in cache_set:
            self._touch_lru(cache_set, tag)
            return None

        # if cache is full, evict according to LRU
        if len(cache_set) >= self.num_ways:
            evicted_tag = cache_set.pop(0)
            evicted_block_addr = (evicted_tag << self.index_bits) | set_index
            evicted_address = evicted_block_addr << self.offset_bits

        cache_set.append(tag)
        return evicted_address

    def invalidate(self, address_int):
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        for i, t in enumerate(cache_set):
            if t == tag:
                cache_set.pop(i)
                break


def run_simulator(config_path, trace_path, output_path):
    with open(config_path, 'r') as f:
        config_line = f.read().strip()
    
    parts = [p.strip() for p in config_line.split(',')]
    line_size = int(parts[0])
    is_inclusive = parts[1].upper() == 'TRUE'
    l1_ways = int(parts[2])
    l1_size = int(parts[3])
    l2_ways = int(parts[4])
    l2_size = int(parts[5])
    
    l1 = Cache("L1", l1_ways, l1_size, line_size)
    l2 = Cache("L2", l2_ways, l2_size, line_size)
    
    output_lines = []
    
    with open(trace_path, 'r') as f:
        trace_lines = f.readlines()
        
    for line in trace_lines:
        line = line.strip()
        line = line.replace(',', ' ')
        tokens = line.split()
            
        addr_str, op = tokens[0], tokens[1]
        addr_int = int(addr_str, 16)
        
        l1_hit = l1.access(addr_int)
        
        if l1_hit:
            # If hit in L1, it updates LRU in L1. It also updates L2 LRU if present.
            _ = l2.access(addr_int)
            result = "L1HIT"
        else:
            l2_hit = l2.access(addr_int)
            if l2_hit:
                result = "L2HIT"
            else:
                result = "MEMACC"
                
        if op == 'R':
            if result == "L2HIT":
                l1_evicted_addr = l1.insert(addr_int)
            elif result == "MEMACC":
                l2_evicted_addr = l2.insert(addr_int)
                l1_evicted_addr = l1.insert(addr_int)
                
                if is_inclusive and l2_evicted_addr is not None:
                    l1.invalidate(l2_evicted_addr)

        output_lines.append(result)
        
    with open(output_path, 'w') as f:
        for res in output_lines:
            f.write(res + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python cache_sim.py <config_file> <trace_file> <output_file>")
        sys.exit(1)
        
    run_simulator(sys.argv[1], sys.argv[2], sys.argv[3])
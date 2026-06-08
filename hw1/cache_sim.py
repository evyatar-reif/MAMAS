import sys
import math
from collections import OrderedDict

class CacheLevel:
    def __init__(self, name, num_ways, total_size, line_size):
        self.name = name
        self.num_ways = num_ways
        self.total_size = total_size
        self.line_size = line_size
        
        # Calculate structure dimensions
        self.num_lines = total_size // line_size
        self.num_sets = self.num_lines // num_ways
        
        # Bit widths [cite: 42]
        self.offset_bits = int(math.log2(line_size))
        self.index_bits = int(math.log2(self.num_sets))
        self.tag_bits = 32 - self.offset_bits - self.index_bits
        
        # Initialize storage: Array of OrderedDicts representing sets
        self.sets = [OrderedDict() for _ in range(self.num_sets)]

    def parse_address(self, address_int):
        """Extracts the set index and tag from a raw 32-bit integer address."""
        block_address = address_int >> self.offset_bits
        set_index = block_address & (self.num_sets - 1)
        tag = block_address >> self.index_bits
        return set_index, tag

    def access(self, address_int, update_lru=True):
        """Checks for a hit. If update_lru is True, refreshes position."""
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        
        if tag in cache_set:
            if update_lru:
                cache_set.move_to_end(tag)  # Mark as most recently used 
            return True
        return False

    def insert(self, address_int):
        """
        Inserts a block tag into its set.
        Returns None if no eviction occurred, or the raw address_int of the
        evicted block if an eviction took place.
        """
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        evicted_address = None

        # Check if already present to update it
        if tag in cache_set:
            cache_set.move_to_end(tag)
            return None

        # Handle eviction if the set is full [cite: 43]
        if len(cache_set) >= self.num_ways:
            # Evict the oldest item (first item in OrderedDict) [cite: 43]
            evicted_tag, _ = cache_set.popitem(last=False)
            
            # Reconstruct an address from the evicted tag and set index to propagate back
            evicted_block_addr = (evicted_tag << self.index_bits) | set_index
            evicted_address = evicted_block_addr << self.offset_bits

        # Add new item
        cache_set[tag] = True
        return evicted_address

    def invalidate(self, address_int):
        """Forces removal of a line if it exists (Used for Inclusive Back-Validation)[cite: 57, 58]."""
        set_index, tag = self.parse_address(address_int)
        cache_set = self.sets[set_index]
        if tag in cache_set:
            del cache_set[tag]


def run_simulator(config_path, trace_path, output_path):
    # 1. Parse Configurations [cite: 19, 20]
    with open(config_path, 'r') as f:
        config_line = f.read().strip()
    
    parts = [p.strip() for p in config_line.split(',')]
    line_size = int(parts[0])
    is_inclusive = parts[1].upper() == 'TRUE'
    l1_ways = int(parts[2])
    l1_size = int(parts[3])
    l2_ways = int(parts[4])
    l2_size = int(parts[5])
    
    # Initialize L1 and L2
    l1 = CacheLevel("L1", l1_ways, l1_size, line_size)
    l2 = CacheLevel("L2", l2_ways, l2_size, line_size)
    
    output_lines = []
    
    # 2. Process Traces [cite: 11, 38]
    with open(trace_path, 'r') as f:
        trace_lines = f.readlines()
        
    for line in trace_lines:
        line = line.strip()
        if not line:
            continue
            
        line = line.replace(',', ' ')
        tokens = line.split()
        if len(tokens) < 2:
            continue
            
        addr_str, op = tokens[0], tokens[1].upper()
        addr_int = int(addr_str, 16)
        
        # Stage 1: Check Cache Status [cite: 47, 53]
        l1_hit = l1.access(addr_int, update_lru=True)
        
        if l1_hit:
            # If hit in L1, it updates LRU in L1. It also updates L2 LRU if present.
            _ = l2.access(addr_int, update_lru=True)
            result = "L1HIT"
        else:
            l2_hit = l2.access(addr_int, update_lru=True)
            if l2_hit:
                result = "L2HIT"
            else:
                result = "MEMACC"
                
        # Stage 2: Perform Allocations/Evictions based on Operation and Result
        if op == 'R':
            if result == "L2HIT":
                # Bring block into L1 [cite: 49]
                l1_evicted_addr = l1.insert(addr_int)
            elif result == "MEMACC":
                # Bring block into both L2 and L1 [cite: 51]
                l2_evicted_addr = l2.insert(addr_int)
                l1_evicted_addr = l1.insert(addr_int)
                
                # Back-invalidation enforcement for Inclusive designs
                if is_inclusive and l2_evicted_addr is not None:
                    l1.invalidate(l2_evicted_addr)
                    
        elif op == 'W':
            # Write-Through / No-Write-Allocate [cite: 44, 45]
            if result == "L1HIT":
                # Data is updated in L1, and written through to L2
                pass 
            elif result == "L2HIT":
                # No Write Allocate -> Data is NOT loaded into L1
                # It is written directly to L2 (and updates L2 LRU, handled in Stage 1) 
                pass
            elif result == "MEMACC":
                # No Write Allocate -> Main Memory updated directly, caches untouched
                pass

        output_lines.append(result)
        
    # 3. Write Output
    with open(output_path, 'w') as f:
        for res in output_lines:
            f.write(res + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python cache_sim.py <config_file> <trace_file> <output_file>")
        sys.exit(1)
        
    run_simulator(sys.argv[1], sys.argv[2], sys.argv[3])
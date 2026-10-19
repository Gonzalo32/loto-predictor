import itertools
import numpy as np

# Test pool 14 from sorteo 3418
pool_14 = [0, 4, 5, 9, 10, 12, 14, 16, 23, 25, 27, 31, 38, 39]
seg_nums = {4, 10, 16, 27, 38} # 5 winning numbers in pool

# If we have 5 numbers in pool 14:
# How many 4-subsets of seg_nums exist?
# seg_nums has 5 numbers: C(5, 4) = 5 quartets:
# {4, 10, 16, 27}, {4, 10, 16, 38}, {4, 10, 27, 38}, {4, 16, 27, 38}, {10, 16, 27, 38}.

# For any ticket of 6 numbers to get 4 hits, it must contain AT LEAST ONE of those 5 quartets!
# In our actual tickets:
# Ticket A: {0, 4, 5, 10, 23, 25} -> only {4, 10} -> 2 hits
# Ticket B: {4, 10, 12, 14, 25, 38} -> only {4, 10, 38} -> 3 hits
# Ticket C: {0, 9, 16, 27, 31, 39} -> only {16, 27} -> 2 hits

# Notice that Ticket B had {4, 10, 38} and Ticket C had {16, 27}!
# If Ticket B had ANY ONE of the 2 numbers that Ticket C took ({16, 27}), Ticket B would have had 4 HITS!
# And Ticket B had {12, 14, 25}. NONE of {12, 14, 25} hit in Segunda!

print("Testing different 3-ticket combinations from pool 14 for 4-hit coverage...")

# What if pool was 10 numbers?
# In pool 10, top 10 numbers were:
# [0, 4, 5, 9, 10, 12, 14, 16, 23, 25] (or similar)
# Let's see what the top 10 numbers were before 3418.

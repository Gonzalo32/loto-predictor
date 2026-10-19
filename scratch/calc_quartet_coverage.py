import itertools

# Let's calculate the coverage of quartets (4-subsets) with 3 tickets of 6 numbers
# Each ticket of 6 has C(6, 4) = 15 quartets.
# 3 tickets can cover at most 3 * 15 = 45 quartets (if completely disjoint quartets).

print(f"{'Pool Size (v)':<15} | {'Total Quartets C(v, 4)':<22} | {'Max Quartets in 3 Tickets':<25} | {'Max Quartet Coverage %'}")
print("="*80)
for v in range(8, 16):
    total_quads = len(list(itertools.combinations(range(v), 4)))
    max_cov = min(45, total_quads)
    pct = (max_cov / total_quads) * 100
    print(f"{v:<15} | {total_quads:<22} | {max_cov:<25} | {pct:.1f}%")

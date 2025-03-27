# Implement a method that can compute c_N for small values of L, at least up to L = 10. This will be useful to validate your Monte Carlo methods.


# Global tracker of s (the set of coordinates visited) and count (the current count of self-avoiding random walks).
s = {(0, 0)}
count = 0


# Compute c_N for a given value of L.
def compute_cn(L):
    global count
    global s
    s = {(0, 0)}
    count = 0
    backtrack(0, 0, L)
    return count


# Backtracks L steps from a coordinate (x, y).
# For each coordinate (a, b) that we have not visited before, backtrack L-1 steps from that new coordinate (a, b).
# L reaches 0 if and only if the walk was self-avoiding, and we add 1 to the number of self-avoiding walks.
def backtrack(x, y, L):
    global count
    global s
    if L == 0:
        count += 1
        return
    for a, b in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if (a, b) in s:
            continue
        s.add((a, b))
        backtrack(a, b, L - 1)
        s.remove((a, b))


# Compute c_N for N = 12.
print(compute_cn(12))

# Implement a more sophisticated Monte-Carlo method that generates a walk of length L by sampling the next step z_i+1 uniformly from the set of possible neighbors of z_i that do not lead to a self-intersection. If there is no such neighbor, which can happen, the walk remains still at z_i until the end of the walk, i.e. z_i+1 = z_i until i = L - 1.


import numpy as np


# Generate self-avoiding walks. (They could still get stuck!)
# In this case, since we want to track all L steps of the walk, we use a list instead of a set to store all previously visited coordinates.
# We filter out coordinates (by checking that they are not in s) to obtain all possible steps.
def generate_self_avoiding_walk(L):
    pos = (0, 0)
    s = [(0, 0)]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(L):
        possible_steps = list(
            filter(
                lambda x: x not in s,
                [(pos[0] + step[0], pos[1] + step[1]) for step in steps],
            )
        )
        # print(f"pos is {pos}")
        # print(f"steps are {possible_steps}")
        if possible_steps:
            pos = possible_steps[np.random.randint(len(possible_steps))]
        s.append(pos)
    # print(s)


# Estimate c_N using importance sampling.

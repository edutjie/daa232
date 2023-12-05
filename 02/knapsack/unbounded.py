import numpy as np


class UnboundedKnapsack:
    def __init__(self, W, val, wt) -> None:
        self.W = W
        self.val = val
        self.wt = wt

    def calculate_max_value(self):
        # dp[i] is going to store maximum
        # value with knapsack capacity i.
        dp = [0] * (self.W + 1)

        # Fill dp[] using above recursive formula
        for i in range(self.W + 1):
            for j in range(len(self.val)):
                if self.wt[j] <= i:
                    dp[i] = max(dp[i], dp[i - self.wt[j]] + self.val[j])

        return dp[self.W]


class BranchBoundUnboundedKnapsack:
    def __init__(self, W, v, w) -> None:
        self.W = W
        self.w = w
        self.v = v
        self.N = list(range(len(w)))
        self.M = None
        self.x_best = None
        self.z_best = 0

    def calculate_max_value(self) -> tuple[int, list[int]]:
        # Procedure 1: Eliminating dominated items
        j = 0
        while j < len(self.N) - 1:
            k = j + 1
            while k < len(self.N):
                wj, wk = self.w[self.N[j]], self.w[self.N[k]]
                vj, vk = self.v[self.N[j]], self.v[self.N[k]]
                if np.floor(wk / wj) * vj >= vk:
                    self.N.pop(k)
                elif np.floor(wj / wk) * vk >= vj:
                    self.N.pop(j)
                    k = len(self.N)
                else:
                    k += 1
            j += 1

        # Procedure 2: Proposed algorithm
        # Initialize
        self.N.sort(key=lambda i: self.v[i] / self.w[i], reverse=True)
        self.v = [self.v[i] for i in self.N]
        self.w = [self.w[i] for i in self.N]

        self.x_best = np.zeros(len(self.N), dtype=int)  # current best solution
        self.z_best = 0  # current best solution value
        x_current = np.zeros(len(self.N), dtype=int)  # current feasible solution

        self.M = np.empty((len(self.N), self.W))

        i = 0
        x1 = int(np.floor(self.W / self.w[i]))
        W_prime = self.W - self.w[i] * x1
        V_N = self.v[i] * x1
        U = self.calculate_U(W_prime, V_N, i)
        m = []
        for i in range(len(self.N)):
            min_mi = min([self.w[j] for j in range(i + 1, len(self.N))], default=np.inf)
            m.append(min_mi)

        # branching
        command = 2
        while True:
            if command == 2:
                i, x_current, W_prime, V_N, command = self.develop(
                    i, x_current, W_prime, V_N, U, m
                )
            elif command == 3:
                i, x_current, W_prime, V_N, command = self.backtrack(
                    i, x_current, W_prime, V_N, m
                )
            elif command == 4:
                i, x_current, W_prime, V_N, command = self.replace(
                    i, x_current, W_prime, V_N, m
                )
            else:
                break

        return self.z_best, self.x_best

    def develop(
        self,
        i,
        x_current,
        W_prime,
        V_N,
        U,
        m,
    ) -> tuple[int, list[int], int, int, int]:
        # Develop
        while True:
            if W_prime < m[i]:
                if self.z_best < V_N:
                    self.z_best = V_N
                    self.x_best = np.copy(x_current)
                    if self.z_best == U:
                        return (i, x_current, W_prime, V_N, 5)
                return (i, x_current, W_prime, V_N, 3)
            else:
                j = self.find_j(W_prime, i)
                U_j = self.calculate_U(W_prime, V_N, j)
                if (V_N + U_j <= self.z_best) or (self.M[i, int(W_prime)] >= V_N):
                    return (i, x_current, W_prime, V_N, 3)
                else:
                    x_current[j] = int(np.floor(W_prime / self.w[j]))
                    V_N += self.v[j] * x_current[j]
                    W_prime -= self.w[j] * x_current[j]
                    self.M[i, int(W_prime)] = V_N
                    i = j

    def backtrack(
        self,
        i,
        x_current,
        W_prime,
        V_N,
        m,
    ) -> tuple[int, list[int], int, int, int]:
        # Backtrack
        while True:
            j = self.find_max_j(x_current, i)
            if j < 0:
                return (i, x_current, W_prime, V_N, 5)
            i = j
            x_current[i] -= 1
            V_N -= self.v[i]
            W_prime += self.w[i]

            if W_prime < m[i]:
                continue
            elif (
                V_N + np.floor(W_prime * (self.v[i + 1] / self.w[i + 1])) <= self.z_best
            ):
                V_N -= self.v[i] * x_current[i]
                W_prime += self.w[i] * x_current[i]
                x_current[i] = 0
            elif W_prime >= m[i]:
                return (i, x_current, W_prime, V_N, 2)
            else:
                return (i, x_current, W_prime, V_N, 4)

    def replace(
        self,
        i,
        x_current,
        W_prime,
        V_N,
        m,
    ) -> tuple[int, list[int], int, int, int]:
        # Replace a jth item with an hth item
        j = i
        h = j + 1
        while True:
            if self.z_best >= V_N + np.floor(W_prime * self.v[h] / self.w[h]):
                return (i, x_current, W_prime, V_N, 3)
            if self.w[h] >= self.w[j]:
                if (
                    self.w[h] == self.w[j]
                    or self.w[h] > W_prime
                    or self.z_best >= V_N + self.v[h]
                ):
                    h += 1
                    continue
                self.z_best = V_N + self.v[h]
                self.x_best = np.copy(x_current)
                x_current[h] = 1
                U_h = self.calculate_U(W_prime, V_N, h)
                if self.z_best == U_h:
                    break
                j = h
                h += 1
            else:
                if W_prime - self.w[h] < m[h - 1]:
                    h += 1
                    continue
                i = h
                x_current[i] = int(np.floor(W_prime / self.w[i]))
                V_N += self.v[i] * x_current[i]
                W_prime -= self.w[i] * x_current[i]
                return (i, x_current, W_prime, V_N, 2)

    def calculate_U(self, W_prime, V_N, i) -> int:
        if i >= len(self.w) - 2:
            return V_N

        v1, v2, v3 = self.v[i], self.v[i + 1], self.v[i + 2]
        w1, w2, w3 = self.w[i], self.w[i + 1], self.w[i + 2]
        z_prime = V_N + (np.floor(W_prime / w2) * v2)
        W_double_prime = W_prime - (np.floor(W_prime / w2) * w2)
        U_prime = z_prime + (np.floor(W_double_prime / w3) * v3)
        U_double_prime = z_prime + (
            np.floor(
                (
                    (W_double_prime + (np.ceil((1 / w1) * (w2 - W_double_prime)) * w1))
                    * (v2 / w2)
                )
                - (np.ceil((1 / w1) * (w2 - W_double_prime)) * v1)
            )
        )

        return max(U_prime, U_double_prime)

    def find_j(self, W_prime, i) -> int:
        j = i + 1
        while j < len(self.w) and self.w[j] > W_prime:
            j += 1
        return j

    def find_max_j(self, x_current, i) -> int:
        j = i if i < len(x_current) else len(x_current) - 1
        while j >= 0 and x_current[j] == 0:
            j -= 1
        return j

n, m = map(int, input().split())
adj = [list(map(int, input().split())) for _ in range(n)]
dp = [[-1]*m for _ in range(n)]
dp[0][0] = adj[0][0]

def f(i, j):
    if i:
        dp[i][j] = max(dp[i][j], dp[i-1][j] + adj[i][j])
    if j:
        dp[i][j] = max(dp[i][j], dp[i][j-1] + adj[i][j])

for i in range(n):
    for j in range(m):
        if dp[i][j] == -1:
            f(i, j)

print(dp[n-1][m-1])

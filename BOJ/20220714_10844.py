MOD = 1_000_000_000
n = int(input())
dp = [[-1]*10 for _ in range(n)]
for i in range(10):
    dp[0][i] = 1
dp[0][0] = 0

def f(i,j):
    if dp[i][j] == -1:
        dp[i][j] = 0
        if j>0:
            dp[i][j] += f(i-1,j-1)
        if j<9:
            dp[i][j] += f(i-1,j+1)
    return dp[i][j]
sum = 0   
for j in range(10):
    sum += f(n-1,j)

print(sum%MOD)


n,k = map(int,input().split())
cnt = 0
coins = [int(input()) for _ in range(n)]
for i in range(n-1,-1,-1):
    if k>=coins[i] and k>=0:
        cnt += (k//coins[i])
        k %= coins[i]
print(cnt)
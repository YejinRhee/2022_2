import sys

input = sys.stdin.readline

N, P = map(int, input().split())
stk = [[] for _ in range(7)]
ans = 0

for _ in range(N):
    line, p = map(int, input().split())

    while stk[line] and stk[line][-1] > p:
        stk[line].pop()
        ans += 1

    if stk[line] and stk[line][-1] == p:
        continue

    stk[line].append(p)
    ans += 1

print(ans)
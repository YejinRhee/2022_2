import sys
sys.setrecursionlimit(10 ** 6)


n, m, k = map(int, input().split())
board = [[True]*n for _ in range(m)]
dy = [1, 0, -1, 0]
dx = [0, 1, 0, -1]


def is_valid(y, x):
    return 0 <= y < m and 0 <= x < n


def dfs(y, x):
    sz = 1
    for i in range(4):
        ny = y + dy[i]
        nx = x + dx[i]
        if is_valid(ny, nx) and board[ny][nx]:
            board[ny][nx] = False
            sz += dfs(ny, nx)
    return sz


for _ in range(k):
    a, b, c, d = map(int, input().split())
    for y in range(a, c):
        for x in range(b, d):
            board[y][x] = False


ans = []
for y in range(m):
    for x in range(n):
        if board[y][x]:
            board[y][x] = False
            ans.append(dfs(y, x))
print(len(ans))
print(*sorted(ans))
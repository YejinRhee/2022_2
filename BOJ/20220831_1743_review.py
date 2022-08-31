from collections import deque
n, m, k = map(int, input().split())
board = [['.']*m for _ in range(n)]
visited = [[False]*m for _ in range(n)]
# board = [['.'*m]*n]  # 이상하게 일케하면 안되드라
# visited = [[False*m]*n] # 이렇게 하면 이상하게 False가 합쳐진다 [[0, 0, 0]] 이러케
for _ in range(k):
    y, x = map(int, input().split())
    board[y-1][x-1] = '#'

dx = [1, 0, -1, 0]
dq = deque()


def is_valid(y, x):
    if 0 <= y and y < n and 0 <= x and x < m:
        return True
    else:
        return False


ans = 0

for y in range(n):
    for x in range(m):
        if not visited[y][x] and board[y][x] == '#':
            visited[y][x] = True
            size = 1
            dq.append((y, x))
            while dq:
                py, px = dq.popleft()
                for i in range(4):
                    ny = py + dy[i]
                    nx = px + dx[i]
                    if is_valid(ny, nx):
                        if not visited[ny][nx] and board[ny][nx] == '#':
                            visited[ny][nx] = True
                            size += 1
                            dq.append((ny, nx))
            ans = max(ans, size)
print(ans)

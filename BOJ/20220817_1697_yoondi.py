'''
최선의 경우를 알고리즘으로 알아내기 어려우니
완탐 => BFS
최악의 경우엔 모든 위치에 다 들르게 되고 각 위치에서는 3군데로 이동 가능하다. 주어지는 범위 크기를 n이라 하면 O(3n)=O(n)
문제에서 최대 10만까지 주어지므로 시간 내에 풀림
'''
from collections import deque

def is_valid_coord(x):
    return 0 <= x < 100_001

def bfs():
    chk = [False] * 100_001
    chk[N] = True
    q = deque()
    q.append((N, 0))
    while q:
        x, d = q.popleft()

        if x == K:
            return d

        nd = d + 1
        for nx in (x + 1, x - 1, 2 * x):
            if is_valid_coord(nx) and not chk[nx]:
                chk[nx] = True
                q.append((nx, nd))


N, K = map(int, input().split())
print(bfs())

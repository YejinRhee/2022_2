# TC 분석 : O( n * m * max(size)) 
from collections import deque
n,m,k = map(int,input().split())
adj = [['.']*m for _ in range(n)] # nm
for _ in range(k):  # k
    py,px = map(lambda x:x-1, map(int,input().split()))
    adj[py][px] = '#'
visited = [[False]*m for _ in range(n)] # nm
visited[0][0] == True
dx = [1,0,-1,0]
dy = [0,1,0,-1]
sizes = []

def is_valid_coord(y,x):
    return 0<=y<n and 0<=x<m

def bfs(q):
    size = 1
    while q: # O(max(size))
        py,px = q.pop()
        visited[py][px] = True
        for i in range(4):  # 4
            ny = py + dy[i]
            nx = px + dx[i]
            if is_valid_coord(ny,nx) and adj[ny][nx] == '#' and visited[ny][nx] == False:
                visited[ny][nx] = True
                q.append((ny,nx))
                size += 1
    return size

# O( n * m * max(size))
for i in range(n):
    for j in range(m): # nm
        if visited[i][j] == False and adj[i][j] == '#': # num(연결요소)
            q = deque([])
            q.append((i,j)) 
            sizes.append(bfs(q))  # O(max(size))
print(max(sizes))
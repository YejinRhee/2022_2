from collections import deque
n,m,k = map(int,input().split())
adj = [['.']*m for _ in range(n)]
for _ in range(k):
    py,px = map(lambda x:x-1, map(int,input().split()))
    adj[py][px] = '#'
visited = [[False]*m for _ in range(n)]
visited[0][0] == True
# print(adj)
dx = [1,0,-1,0]
dy = [0,1,0,-1]
sizes = []

def is_valid_coord(y,x):
    return 0<=y<n and 0<=x<m

def bfs(q):
    size = 1
    while q:
        py,px = q.pop()
        visited[py][px] = True
        # print("q starts from : ({},{})".format(py,px))
        for i in range(4):
            ny = py + dy[i]
            nx = px + dx[i]
            if is_valid_coord(ny,nx) and adj[ny][nx] == '#' and visited[ny][nx] == False:
                visited[ny][nx] = True
                q.append((ny,nx))
                # print("q.append(({},{}))".format(ny,nx))
                size += 1
    # print(size)
    return size
    
for i in range(n):
    for j in range(m):
        if visited[i][j] == False and adj[i][j] == '#':
            q = deque([])
            q.append((i,j)) # py,px
            sizes.append(bfs(q))
print(max(sizes))
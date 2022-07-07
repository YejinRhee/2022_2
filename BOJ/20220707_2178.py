from collections import deque
n,m = map(int,input().split())
adj = [list(input()) for _ in range(n)]
q = deque()
q.append((0,0,1)) # py,px,d
visited = [[False]*m for _ in range(n)]
visited[0][0] = True
dx = [1,0,-1,0]
dy = [0,1,0,-1]

def is_valid_coord(y,x):
    return 0<=y<n and 0<=x<m

def bfs():
    while q : 
        py,px,d = q.popleft()      
        if py == n-1 and px == m-1:
            return d
        nd = d+1
        for i in range(4):
            ny = py+dy[i]
            nx = px+dx[i]
            if is_valid_coord(ny,nx) and adj[ny][nx] == '1' and visited[ny][nx] == False:
                visited[ny][nx] = True
                q.append((ny,nx,nd))
    print("empty queue")

print(bfs())

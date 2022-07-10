from collections import deque
n,m = map(int, input().split())
adj = [list(input()) for _ in range(n)]
visited = [[False]*m for _ in range(n)]
q =deque()
q.append((0,0,1))
visited[0][0] = True
dx = [1,-1,0,0]
dy = [0,0,1,-1]

def is_valid(y,x):
    return 0<=y<n and 0<=x<m

def bfs():
    while q:
        py,px,d = q.popleft()       
        if py == n-1 and px == m-1:
            return d
        nd = d+1
        print(nd)
        
        for i in range(4):
            nx = px + dx[i]
            ny = py + dy[i]
            if is_valid(ny,nx) and adj[ny][ny] =='1' and visited[ny][nx]==False:
                visited[ny][nx]= True
                q.append((ny,nx,nd))
                
print(bfs())     
        
    

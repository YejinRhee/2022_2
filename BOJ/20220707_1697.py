from collections import deque
x,k = map(int,input().split())

q = deque()
q.append(x) 
position = [0]*(10**5)
opt = ['+','*','-']
def bfs():
    while q:
        px = q.popleft()
        if px == k:
            print(position[px])
            break
        for i in range(3):   # 이렇게 가능. for nx in (px-1, px+1, px*2)
            if opt[i] == '+':
                nx = px+1
            elif opt[i] == '*':
                nx = 2*px
            else:
                nx = px - 1
            
            if 0<=nx<=10**5 and position[nx]==0:       
                position[nx] = position[px] + 1
                q.append(nx)

bfs()












# position = [0]*(10**5)
# print (position)
# times = []
# opt = ['+','*','-']
# def bfs(q):
#     t=0
#     while q:
#         px= q.pop()
#         position[px] = t
#         if px == k:
#             print(t)
#             times.append(t)
#         for i in range(3):
#             if opt[i] == '+':
#                 nx = px+1
#             elif opt[i] == '*':
#                 nx = 2*px
#             else:
#                 nx = px - 1
                
#             if 0 <=nx <= 10**5 and not position[nx] : 
#                 q.append((nx,nt))
# print (position)
# print("min(times) : ",min(times))
        

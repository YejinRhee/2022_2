from bisect import bisect_left, bisect_right
n,m = map(int,input().split())
s = []
s_init = []
for _ in range(n) : 
    tmp = input()
    s_init.append(tmp[0])
    s.append(tmp)
t = [input() for _ in range(m)]
s.sort()
s_init.sort()
t.sort()
cnt = 0

for i in range(m):
    l = len(t[i])
    idx = bisect_left(s_init,t[i][0])
    num_check = bisect_right(s_init,t[i][0]) - bisect_left(s_init,t[i][0])
    yes_prefix = False
    for j in range(num_check): 
        for x in range(l):
            if s[idx+j][x] != t[i][x]:
                break
            if x == l-1:
                yes_prefix = True
        if yes_prefix:
            cnt += 1 
            break
print(cnt)    

numtc, nump = map(int,input().split())
stk = [[],[],[],[],[],[]]
cnt = 0
for _ in range(numtc):
    s,p = map(int,input().split())
    s -= 1
    if len(stk[s]) == 0:
        stk[s].append(p)
        cnt +=1 
    else:
        if stk[s][-1] == p:
            pass
        elif stk[s][-1] < p:
            stk[s].append(p)
            cnt += 1
        else: # 보다 크면
            while len(stk[s]) != 0 and stk[s][-1] > p:
                stk[s].pop()
                cnt += 1
            if len(stk[s]) == 0:
                stk[s].append(p)
                cnt += 1
                pass
            elif stk[s][-1] == p:
                pass
            elif stk[s][-1] < p:
                stk[s].append(p)
                cnt += 1
                
print(cnt)

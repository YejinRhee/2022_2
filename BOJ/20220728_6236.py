n,m = map(int,input().split())
plans = [int(input() for _ in range(n))]

hi = sum(plans)
lo = max(plans)-1

def is_possible(m):
    cnt = 0
    money = 0
    for plan in plans:
        if money<plan:
            cnt += 1
            money = m
        money -= plan
    return cnt<=m

while lo+1<hi:
    mid = (lo+hi)//2
    if is_possible(mid):
        hi = mid
    else:
        lo = mid
        
print(hi)
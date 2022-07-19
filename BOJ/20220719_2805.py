n,m = map(int,input().split())
trees = list(map(int,input().split()))
hi = max(trees)
lo = 0  # 이게 왜 0이어야하는지 아니 ...? 그건 바로 input : 1 1 \n 1 인 경우에 대비해서,,
mid = (lo+hi)//2

def calculate_lumber(mid):
    sum_length = 0
    for tree in trees:
        if tree > mid:
            sum_length += (tree-mid)
    return sum_length

while lo+1<hi:
    if m <= calculate_lumber(mid):
        lo = mid   # 범위는  [mid,hi]
    else :
        hi = mid   # 범위는  [lo, mid]
    mid = (lo+hi)//2

print(lo)
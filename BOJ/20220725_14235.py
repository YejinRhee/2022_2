import heapq
gift = []
numtc = int(input())

for _ in range(numtc):
    a = list(map(int, input().split()))
    if a[0] == 0:
        if len(gift)==0:
             print("-1")
        else:
            tmp = -heapq.heappop(gift)
            print(tmp)
    else:
        for i in range(a[0]):
            heapq.heappush(gift,-a[i+1])

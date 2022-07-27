# heapq에 대한 몰랐던 특성을 배움과 동시에
"""
heapq에 튜플로 (절댓값, 원본값) 쌍을 지어 노드에 넣어주면 
1차적으로 절댓값이 가장 작은 노드를, 
만약 절댓값이 같다면 2차적으로 원본값이 작은 노드를 
더 우선순위를 높게 쳐서 루트노드로 올림
"""
# python STL의 치명적인 단점을 알 수 있었던 문제
"""
STL은 간편하지만, 내가 원하는 기능을 자유자재로 구현하기에는 제약이 있다.
ex. 절댓값이 같은 애들을 모두 nsmallest()로 찾았다 하더라도, 게 중 원하는 key 값 갖는 애를
탐색해서 pop해주는 기능은 없다..
"""

# yoondi's code
import heapq as hq, sys

input = sys.stdin.readline
min_heap = []
for _ in range(int(input())):
    x = int(input())
    if x:
        hq.heappush(min_heap, (abs(x), x))
    else:
        print(hq.heappop(min_heap)[1] if min_heap else 0)

# my code
import heapq as hq
tups = []
num_abs_val = [[0]*(pow(2,31)+1)]
n = int(input())
for _ in range(n):
    value = int(input())
    abs_val = abs(value)
    if value == 0:
        if len(tups) == 0:
            print(0)
        else:
            if num_abs_val[abs_val] == 1:
                temp = hq.heappop(tups)
                print(temp[1])
            else:
                templist = [hq.nsmallest(num_abs_val[abs_val], tups, key = tups[1])]
                if -tups[0] in templist:
                    
    else: 
        hq.heappush(tups,(abs_val,value))
        num_abs_val[abs_val] += 1
    
    


hq.heappush(tups,(1,'a'))
hq.heappush(tups,(0,'b'))
hq.heappush(tups,(1,'c'))
hq.heappush(tups,(2,'d'))
print(tups)

hq.nsmallest(tups,key=str.lower)

from itertools import combinations

tc = 0
while True:
    # a = list(map(int,input().split()))
    k, *S = map(int,input().split())  # 나머지를 S에 담겠다는 뜻
    # 파이썬에서 asterik의 쓰임새가 여러가지가 있으니 구글링해보삼 
    if k==0:
        break
    
    if tc:
        print()
        
    for combi in combinations(S,6): # 애초에 입력이 오름차순이라, 굳이 정렬 안해도 돼
        print(*combi)
    
    tc += 1 
    






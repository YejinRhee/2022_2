prefixes = set()
N, M = map(int, input().split())

for _ in range(N):
    word = input()
    
    for i in range(1, len(word) + 1):
        prefixes.add(word[:i])

print(sum(1 for _ in range(M) if input() in prefixes))


'''
N개의 문자열 각각에 대해 존재하는 모든 접두사를 집합에 넣어두고 검사해야 하는 문자열이 주어질 때 마다 집합에 있는지 조회한다. 집합의 조회는 $O(1)$ 이므로 M번 반복하면 시간 복잡도 $O(M)$ 일 것이고, 접두사가 많을테니 접두사 개수가 몇개까지 있을 수 있는지를 따져보자.

문자열 길이가 최대 L 이라면 접두사도 L개 존재한다. N개의 문자열 각각에 대해 전부 만들면 총 N * L 개의 접두사가 나온다. N * L ≤ 10,000 * 500 = 5,000,000 이다.

최종적으로 존재할 수 있는 접두사를 전부 만들어 집합에 넣고, M번 각각 집합에 존재여부를 조회하는 시간 복잡도는 $O(NL+M)=O(NL)$ 로 제한 시간 내에 통과한다.
'''
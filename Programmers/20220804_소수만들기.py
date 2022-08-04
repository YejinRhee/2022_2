from itertools import combinations

def is_prime(elem):
    answer = True
    for i in range(2,elem//2):
        if elem % i == 0:
            answer = False
            # print(elem, i, "nope!")
            break
    return answer 

def solution(nums):
    answer = 0
    lst = list(combinations(nums,3))
    # print(lst)
    lst2 = []
    for i in range(len(lst)):
        lst2.append(sum(lst[i]))
    # print(lst2)
    for elem in lst2:
        if is_prime(elem):
            answer += 1

    return answer
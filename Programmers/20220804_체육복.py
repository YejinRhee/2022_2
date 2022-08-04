def solution(n, lost, reserve):
    answer = 0
    num_clothe = [1]*n
    print(num_clothe)
    for l in lost:
        num_clothe[l-1] = 0
    for r in reserve:
        num_clothe[r-1] += 1
    print(num_clothe)
    
    for r in reserve:
        idxr = r-1
        if idxr > 0 :
            if num_clothe[idxr-1] == 0:
                num_clothe[idxr-1] += 1
                num_clothe[idxr] -= 1
                
        if idxr < len(num_clothe)-1 :
            if num_clothe[idxr+1] == 0:
                if num_clothe[idxr] > 0 :
                    num_clothe[idxr+1] += 1
                    num_clothe[idxr] -= 1
    for num in num_clothe:
        if num > 0:
            answer +=1
    
    return answer
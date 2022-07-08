MOD = 1_000_000_000
n = int(input())
cache = [[-1]*10 for _ in range(n+1)]  # cache[i] = [f(i,0),f(i,1), ... , f(i,9)] 
cache[1][0] = 0
for i in range(1, 10):
    cache[1][i] = 1

def f(n,x):
    if cache[n][x] == -1: # 방문 안했냐
        cache[n][x] = 0 # 방문 했다
        # 관건은 f(n-1)의 마지막 자리 수 가지고 x 만들기
        if x > 0 :
            cache[n][x] += f(n-1,x-1) # f(n-1,x-1)의 마지막값+1으로도 x가 될 수 있고
        if x < 9:
            cache[n][x] += f(n-1,x+1) # f(n-1,x+1)의 마지막값-1으로도 x가 될 수 있지 
    
    return cache[n][x]

sum = 0
for x in range(10):
    sum += f(n, x)
print(sum % MOD)
# for i in cache:
#     print(i)




# ~~ my trial ~~ 
# def f(n, x):
#     if cache[n][x] == -1:
#         if x == 0 or x == 9:
#             if n == 1: # 짜피 n은 1 이상임
#                 pass
#             # elif n == 2 and x == 0:
#             #     cache[n][x] = 1
#             else:
#                 cache[n][x] = f(n-1, x)
#         else:
#             cache[n][x] = 2 * f(n-1, x)
#     else:
#         pass # 이미 메모이제이션 했음 
#     return cache[n][x]





# print("cache[{}}][{}] : ".format(n,x), cache[n][x])


# n = int(input())
# if n == 3:
#     print("pass")
#     pass
# else:
#     print("why print this")
# print("done")
# nCr을 10007로 나눈 나머지
MOD = 10_007
n, k = map(int, input().split())
if k == 0:
    print(1)
    exit()
# nCr값을 cache matrix에 저장하는 방법도 타당하지
# bino(n,r) = bino(n-1,r-1)+bino(n-1,r)을 적용할거라면.
# but if bino(n,r) = n!/(n-r)!r!을 사용하고싶다면
# 1차 array인 chache에, factorial값을 저장하는 방법으로도 가능해
# cache[n-1] = n! = cache[n-2]*n

cache = [0]*(n+1)
cache[0] = 1
cache[1] = 1
# cache[n] = (n+1)!
for i in range(2, n+1):
    cache[i] = (i)*cache[i-1]\

def f(x):
    if x < 0:
        return cache[0]
    return cache[x]

print("f({}) : \n".format(n), f(n))
print("f({}) : \n".format(k), f(k))
value = f(n)/(f(n-k) *f(k))
# print(value% MOD)
# value = (f(n) % MOD)/((f(n-k) % MOD)*(f(k) % MOD))
# print(value)
print(int(value % MOD))
# if (f(n) > (f(n-k) *f(k))):
#     print("f(n) > (f(n-k) * f(k))")
# else : 
#     print("f(n) <= (f(n-k) * f(k))")

# def factorial(x):
#     if x==1:
#         return factorial[x-1]
#     return x * factorial(x-1)

# def f(x):
#     if x<=0 :
#         return cache[0]
#     return cache[x-1]
# print("f(n) : ", f(n))
# print("f(n-k): ", f(n-k))
# print("f(k): ", f(k))
# value = f(n)/(f(n-k)*f(k))

# value = cache[n-1]/(cache[n-k-1]*cache[k-1])

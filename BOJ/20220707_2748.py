n = int(input())
fibs = [0,1]
for i in range(1,n+1):
    fib = fibs[i-1]+fibs[i]
    fibs.append(fib)
print(fibs[n])
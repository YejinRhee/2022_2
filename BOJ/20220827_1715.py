n = int(input())
lst = [int(input()) for _ in range(n)]
lst.sort(reverse = True)
temp = lst.pop()
sum = 0
for _ in range(n-1):
  temp += lst.pop()
  sum += temp
  temp = sum

print(sum)

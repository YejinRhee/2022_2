# deepcopy한 순간부터 주소가 달라짐 !
import copy
a = [[1,2],[3,4]]
b = copy.deepcopy(a)
print(a) # [[1, 2], [3, 4, 5]]
print(b) # [[1, 2], [3, 4]]
print(id(a)) # 1700928108864
print(id(b)) # 1700928109120

a[1].append(5)
print("appended a : ", id(a)) # 1700928108864

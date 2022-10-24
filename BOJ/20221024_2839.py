#n = int(input())
# sum = 0
# n_5 = 0
# n_3 = 0
# while sum <= n :
#     sum += 5
#     n_5 += 1
#
# while n_5 >= 0 :
#     if sum == n:
#         print(n_5)
#         exit()
#     else:
#         sum -= 2
#         n_5 -= 1
#         n_3 += 1
#         while sum < n :
#             sum += 3
#             n_3 += 1
# if sum <= 0:
#     print(-1)
#

sugar = int(input())

bag = 0
while sugar >= 0 :
    if sugar % 5 == 0 :  # 5의 배수이면
        bag += (sugar // 5)  # 5로 나눈 몫을 구해야 정수가 됨
        print(bag)
        break
    sugar -= 3
    bag += 1  # 5의 배수가 될 때까지 설탕-3, 봉지+1
else :
    print(-1)


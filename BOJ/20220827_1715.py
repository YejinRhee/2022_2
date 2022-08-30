'''
묶음 시행 횟수는 고정인데, 어느 카드끼리 묶어야 할지 순서를 골라야 한다. 순열 경우의 수이므로 N! 가지가 존재한다. N ≤ 100,000 이니 이런 완전탐색으로는 시간 초과가 발생한다
. 가장 작은 두 카드 묶음을 찾아 이들을 묶어주고, 다시 남은 모든 카드 묶음들 중 가장 작은 두 카드 묶음을 찾아 이들을 묶어주고, … 반복하면 된다. 
‘항상 제일 두 작은 카드 묶음을 찾아 이들을 묶는다’는 그리디 전략이 먹힌다.
'''


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

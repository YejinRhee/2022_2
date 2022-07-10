MOD = 1_000_000_007

n, m = map(int, input().split())
board = [list(map(int, input().split())) for _ in range(n)]
dp = [[0]*m for _ in range(n)]

# 이거 굳이 안써줘도 잘 돌아가서 주석처리
# if n == 1:
#     print(sum(board[0]))
#     exit()

# if m == 1:
#     for i in range(n):
#         if board[i][0] == 0:
#             print(0)
#             exit()
#     print(1)
#     exit()

for i in range(m):
    dp[n-1][i] = board[n-1][i]


def is_valid_coord(py, px):
    return 0 <= py < n and 0 <= px < m


cnt = 0
for i in range(n-1, -1, -1):
    for j in range(m):
        ny = i-1
        for nx in (j-1, j, j+1):
            if is_valid_coord(ny, nx) and board[i][j]:
                if board[ny][nx] == 1:
                    if ny == 0:
                        cnt += dp[i][j]
                    else:
                        dp[ny][nx] += dp[i][j]
                else:
                    pass  # dp[ny][nx] = 0 라는 뜻

# for row in dp:
#     print("row : ",*row)
print(cnt)

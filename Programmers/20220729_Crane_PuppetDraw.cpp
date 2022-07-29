#include <string>
#include <vector>
#include <stack>
#include <iostream>

using namespace std;

int solution(vector<vector<int>> board, vector<int> moves) {
	int answer = 0;
	stack<int> bin;
	for (int move : moves) {
		int i = move-1;
		int crane;
		for (int j = 0; j < board.size(); j++) {
			if (board[j][i] != 0) { // j가 마지막 아니거나 board[i][j] != 0인 경우 
				crane = board[j][i];
				board[j][i] = 0;
				if (bin.empty() == 0 && bin.top() == crane){
						answer+=2;
						bin.pop();
				}
				else
					bin.push(crane);
				break;
			}
			else if (j == board.size() - 1 && board[j][i] == 0) {
				break;
			}
		}
	}
	return answer;
}

int main() {
	vector<vector<int>> board = { {0, 0, 0, 0, 0},{0, 0, 1, 0, 3},{0, 2, 5, 0, 1},{4, 2, 4, 4, 2},{3,5,1,3,1} };
	vector<int> moves = { 1,5,3,5,1,2,1,4 };
	int answer = solution(board, moves);
	cout << answer;
}


/*
0, 0, 0, 0, 0
0, 0, 1, 0, 3
0, 2, 5, 0, 1
4, 2, 4, 4, 2
3, 5, 1, 3, 1

*/
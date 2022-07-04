n,m = map(int,input().split())
boxes = list(map(int,input().split()))
kids = list(map(int,input().split()))

for i in range(m):
    boxes.sort()
    if boxes[0]>kids[i]:
        boxes[0] -= kids[i]
    else:
        print(0)
        exit()
print(1)





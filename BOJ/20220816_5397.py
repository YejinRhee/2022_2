numtc = int(input())

def tellmepw():
    original = input()
    stk1,stk2 = [],[]
    for cha in original:
        if cha == '<' :
            if stk1 :
                stk2.append(stk1.pop())
        elif cha == '>':
            if stk2:
                stk1.append(stk2.pop())
        elif cha == '-':
            if stk1:
                stk1.pop()
        else:
            stk1.append(cha)
    for _ in range(len(stk2)):
        stk1.append(stk2.pop())
    return stk1

for _ in range(numtc):
    pw = tellmepw()
    print(*pw,sep='')

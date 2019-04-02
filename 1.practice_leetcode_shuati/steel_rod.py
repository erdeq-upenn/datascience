def CutRod(p, n):
    if n == 0:
        return 0
    q = -1
    for i in range(1, n+1):
        q = max(q, p[i] + CutRod(p, n-i))
        # '''
        # tmp = p[i] + CutRod(p, n-i)
        # if q < tmp:
        #     q = tmp
        # '''
    return q
p=[0,1,5,8,9,10,17,17,20,24,30]
print("Max_profit",CutRod(p,4))


def BottomUpCutRod(p, n):
    r = [0]*(n+1)
    for i in range(1, n+1):
        if n == 0:
            return 0
        q =0
        for j in range(1, i+1):
            q = max(q, p[j]+r[i-j])
            r[i] = q
    return r[n],r
p=[0,1,5,8,9,10,17,17,20,24,30]
print(BottomUpCutRod(p, 10))   #  (30, [0, 1, 5, 8, 10, 13, 17, 18, 22, 25, 30])

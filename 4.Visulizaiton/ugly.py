#ugly number
def isUgly(num):
    """
    :type num: int
    :rtype: bool
    """
    for p in 2, 3, 5, 7, 11, 13, 17:
        while num % p == 0 < num:
            num /= p
    return num == 1
for i in range(100):
    if isUgly(i)==True:
        print (i, 'is Ugly')

#counting-sort algo
#Dequan

def counting_sort(array, maxval):
    """in-place counting sort"""
    m = maxval + 1
    count = [0] * m               # init with zeros
    for a in array:
        count[a] += 1             # count occurences
    i = 0
    for a in range(m):            # emit
        for c in range(count[a]): # - emit 'count[a]' copies of 'a'
            array[i] = a
            i += 1
    return (array,count)

arr= [1,3,5,2,2,4,3]
print('before sort',arr)
sorted_arr,count=counting_sort(arr,max(arr))
print('count',count)
print('after sort',sorted_arr)
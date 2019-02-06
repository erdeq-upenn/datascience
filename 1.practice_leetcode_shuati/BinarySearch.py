# practice code of BinarySearch

# Return -1 if not found, return index of target x in arr present
def BinarySearch(arr,l,r,x):


    #check base case
    if r >=1:

        mid = 1 + (r-1)/2

        #if lement is at the middle
        if arr[mid]==x:
            return mid

        elif arr[mid]>x:
            BinarySearch(arr,l,mid-1,x)

        elif arr[mid]<x:
            BinarySearch(arr,mid+1,r,x)
        else:
            return -1

#test arry
a = [1, 4, 8, 10, 20, 34, 40]
x = 4
#function call
result = BinarySearch(a, 0, len(a)-1, x)

if result != -1:
    print "Element is present at index %d" % result
else:
    print "Element is not present in array"

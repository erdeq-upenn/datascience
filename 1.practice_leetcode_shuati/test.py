import re

string ="A1.45, b5, 6.45, -$4.9"
foo = [1,2,3,'a',None,(),[],]
print len(foo)
word = "positive"
print word[:2]

print re.findall(r"\d+\.?\d*",string)
a = 2<<2
print a

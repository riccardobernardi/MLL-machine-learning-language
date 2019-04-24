# a = [1,2,3,4]
# b = [1,2,3,4]
#
# for i in range(0,a.__len__()):
#     print(a[i])
#     if a[i]==2:
#         a.insert(2,10)
#
# print()
#
# for i in b:
#     print(i)
#     if i==2:
#         b.insert(2,10)


# def add(l:list,addBefore:object,match:object,addAfter:object) -> list:
#     for i in l:
#         print("in loop")
#         if i==match:
#             l.insert(l.index(match)-1,addBefore)
#             l.insert(l.index(match)+1, addAfter)

c = [1,2,3,4]
#add(c,{'0':0},3,{'1':1})

# for i in c:
#     print(i)



# class IterVector:
#     def __init__(self, array : list):
#         self.array:list = array
#         self.matched:int = 0
#
#     def match(self,o : object):
#         for i in self.array:
#             if i==o:
#                 self.matched = self.array.index(i)
#                 return self
#
#     def add_before(self,o : object):
#         self.array.insert(self.matched,o)
#         return self
#
#     def add_after(self,o : object):
#         self.array.insert(self.matched+1,o)
#         return self
#
#     def print(self):
#         for i in self.array:
#             print(str(i) + " ",end='')
#         print()
#         return self
#
#     def get(self):
#         return self.array
#
# def MatchSub(self,l:list,b:object,c:object,a:object):
#     return IterVector(l).match(c).add_after(a).add_before(b).print()
#
# iv = IterVector(c).match(3).add_after(1).add_before(-1).print().get()
# disp=0
# r = range(0,len(c))
#
# for i in r:
#     print(c[i])
#     if i==1:
#         c.insert(2,10)
#         r = range(0,len(c))
#
# for i in range(0,len(c)):
#     print(c[i])
#
#
# print(range(0,10))

i=0
while True:
    if i == len(c):
        break
    else:
        print(c[i])
        if i==1:
            c.insert(2,10)
        i += 1

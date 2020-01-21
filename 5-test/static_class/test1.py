class Dates:
    def __init__(self, date):
        self.date = date

    def getDate(self):
        return self.date
    def initDate(self):
        return ("15-13-2016")
    @staticmethod
    def toDashDate(date):
        return date.replace("/", "-")
date = Dates("15-12-2016")
dateFromDB = "15/12/2016"
dateWithDash = Dates.toDashDate(dateFromDB)
if(date.getDate() == dateWithDash):
    print("Equal")
else:
    print("Unequal")

# ============


class Kls(object):
    num_inst = 0

    def __init__(self):
        Kls.num_inst = Kls.num_inst + 1

    def p_inst(self):
        print(self.num_inst)

    @classmethod
    def get_no_of_instance(cls):
        return cls.num_inst


ik1 = Kls()
ik2 = Kls()
ik1.p_inst()  # test

print(ik1.get_no_of_instance())
print(Kls.get_no_of_instance())


"""
在上述例子中，我们需要统计类Kls实例的个数，因此定义了一个类变量num_inst来存放实例个数。
通过装饰器@classmethod的使用，方法get_no_of_instance被定义成一个类方法。在调用类方法时，
Python 会将类（class Kls）传递给cls，这样在get_no_of_instance内部就可以引用类变量num_inst。
由于在调用类方法时，只需要将类型本身传递给类方法，因此，既可以通过类也可以通过实例来调用类方法。

"""
# ==============

IND = 'ON'


class Kls(object):
    """
    在代码中，我们定义了一个全局变量IND，由于IND跟类Kls相关，所以我们将方法checkind放置在类Kls中定义。
    方法checkind只需检查IND的值，而不需要引用类或者实例，因此，我们将方法checkind定义为静态方法。
    对于静态方法，Python 并不需要传递类或者实例，因此，既可以使用类也可以使用实例来调用静态方法
    """
    def __init__(self, data):
        self.data = data

    @staticmethod
    def checkind():
        return IND == 'ON'

    def do_reset(self):
        if self.checkind():
            print('Reset done for: %s' % self.data)
        else:
            print('DB is off for %s' % self.data)

    def set_db(self):
        if self.checkind():
            print('DB connection made for: %s' % self.data)


ik1 = Kls(24)
ik1.do_reset()
ik1.set_db()


#  =============
class Kls(object):
    def foo(self, x):
        print('executing foo(%s,%s)' % (self, x))

    @classmethod
    def class_foo(cls,x):
        print('executing class_foo(%s,%s)' % (cls,x))

    @staticmethod
    def static_foo(x):
        print('executing static_foo(%s)' % x)


ik = Kls()

# 实例方法
ik.foo(1)
print(ik.foo)
print('==========================================')

# 类方法
ik.class_foo(1)
Kls.class_foo(1)
print(ik.class_foo)
print('==========================================')

# 静态方法
ik.static_foo(1)
Kls.static_foo('hi')
print(ik.static_foo)
"""
对于实例方法，调用时会把实例ik作为第一个参数传递给self参数。因此，调用ik.foo(1)时输出了实例ik的地址。

对于类方法，调用时会把类Kls作为第一个参数传递给cls参数。因此，调用ik.class_foo(1)时输出了Kls类型信息。
前面提到，可以通过类也可以通过实例来调用类方法，在上述代码中，我们再一次进行了验证。

对于静态方法，调用时并不需要传递类或者实例。其实，静态方法很像我们在类外定义的函数，只不过静态方法可以通过类或者实例来调用而已。

值得注意的是，在上述例子中，foo只是个函数，但当调用ik.foo的时候我们得到的是一个已经跟实例ik绑定的函数。调用foo时需要两个参数，但调用ik.foo时只需要一个参数。foo跟ik进行了绑定，因此，当我们打印ik.foo时，会看到以下输出:

<bound method Kls.foo of <__main__.Kls object at 0x0551E190>>
1
当调用ik.class_foo时，由于class_foo是类方法，因此，class_foo跟Kls进行了绑定（而不是跟ik绑定）。当我们打印ik.class_foo时，输出：

<bound method type.class_foo of <class '__main__.Kls'>>
1
当调用ik.static_foo时，静态方法并不会与类或者实例绑定，因此，打印ik.static_foo（或者Kls.static_foo）时输出：

<function static_foo at 0x055238B0>
1
概括来说，是否与类或者实例进行绑定，这就是实例方法，类方法，静态方法的区别。

"""

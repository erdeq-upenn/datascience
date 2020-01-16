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

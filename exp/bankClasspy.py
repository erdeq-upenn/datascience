# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:19:50 2019

@author: Dequan Er
"""

class Bank:
    # Simulate banks
    def __init__(self):
        self.balance = 0
    def deposite(self,amount):
        self.balance = self.balance + amount
        
    def withdraw(self,amount):
        self.balance = self.balance - amount
        
    def getbalance(self):
        return self.balance
    
myacc = Bank()
myacc.deposite(10)
print(myacc.getbalance())


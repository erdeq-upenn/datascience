# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:35:40 2019

@author: Dequan Er
"""

class bank():
    def __init__(self):
        self.balance = 0
    def deposit(self,amount):
        self.balance = self.balance + amount
        
    def showbalance(self):
        return self.balance
    
mya = bank()

mya.deposit(10)
print(mya.showbalance())
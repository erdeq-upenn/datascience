# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:59:08 2019

@author: Dequan Er
"""

# this is a practice of creating a class
# Classes provide a means of bundling data and functionality together. 
class bank():
    def __init__(self,amount):
        bank.val = amount  # This is data type
    def deposit(self,amount):
        self.val = self.val + amount  # This is function type
    def withdraw(self,amount):
        self.val = self.val - amount
    def show(self):
        return self.val
        
x = bank(100)
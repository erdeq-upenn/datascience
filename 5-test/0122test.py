#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:03:11 2020

@author: Dequan
"""

from __future__ import print_function
import torch

x = torch.rand(5, 3)
print(x)

y = torch.ones(5, 3, dtype=torch.float)
print(x.size())

y.add_(x)
print(x, y)

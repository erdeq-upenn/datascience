# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:47:48 2019

@author: Dequan
"""

import pdfkit
import os


def convert_html2pdf(f):
    #convert html to pdf file 
    #input filename.html
    #output filename.pdf
    f_name=f.split('.')
    output = fname[-2]+'.pdf'
    print(f_name,output)
    pdfkit.from_file(f, output)
#pdfkit.from_string('Hello!', 'out.pdf')

flist = os.listdir();
for f in flist:
    fname = f.split('.')
    if fname[-1]=='html':
#        print(fname)
        convert_html2pdf(f)
        

# enter.py - code for replicating build-in Matlab input Function
# Code written by Nabanita Sarkar
# Input format: 
# Single Integer or float
#    Eg: 5
#              
#   Integers or floats seperated by comma
#   Eg: 8,9
#              
#   List of integers or floats
#   Eg: [6,7]
    
    
def enter(promt = True):
    str = input(promt)
    str1 = str.replace('[','')
    str2 = str1.replace(']','')

    list = str2.split(",")
    
    li = []
    for i in list:
        li.append(float(i))
    return li

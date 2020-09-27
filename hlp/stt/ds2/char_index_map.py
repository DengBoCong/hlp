# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:50:12 2020

@author: 彭康
"""

#字符集构建的字典char_map,index_map

char_map_str = """
<SPACE> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
' 28
"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch

index_map[1] = ' ' #在index_map里边以1:' '为键值对而char_map里边以'<space>':1为键值对.

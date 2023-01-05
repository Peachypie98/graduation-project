#!/usr/bin/env python3

def h3_height(x):
    if x == 50:
        R = 2
    else:
        x = round(x,1)
        R = (2*x)/50 

    # Basic Formula --> h3 = h1 + (d * (h1-h2)) / (D-R)
    # 7cm은 감자의 절단면까지의 높이
    # Design Attributes
    h1 = 19.8
    h2 = 5.3 + 5.4
    d = 7.8
    D = 12.6

    full_h3 = h1 + ((h1-h2)*d/(D-R))
    h3 = round(full_h3 - 7.4, 1)
    
    return h3


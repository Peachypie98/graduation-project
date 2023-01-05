#!/usr/bin/env python3
import numpy as np
import cv2 as cv

def radius(frame):
    image = frame
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for i in range(480):
        for j in range(640):
                if int(image[i,j,0])>41 and int(image[i,j,1])>45 and int(image[i,j,2])>14:
                    image[i,j,0] = 200; image[i,j,1] = 200; image[i,j,2] = 200

    # convert image to canny
    edges = cv.Canny(image, 30, 100)
    edges = np.array(edges)

    # crop image to 100-100
    image = edges[190:290, 270:370]

    a, b, c, d = [0, 0], [0, 0], [0, 0], [0, 0]

    # algorithm for finding radius 1
    for j in range(83):
        if j>14:
            for i in range(83):        
                if image[i,j] == 255:
                    a = [i,j]  
                    break
                else:
                    continue            
            if a!=[0,0]:
                break
            else:
                continue
        else:
            continue


    for j in range(83):
        if j>12:
            for i in range(83):        
                if image[i,82-j] == 255:
                    c = [i,82-j]  
                    break
                else:
                    continue            
            if c!=[0,0]:
                break
            else:
                continue
        else:
            continue


    for i in range(83):
        if i>25:
            for j in range(83):       
                if image[i,j] == 255:
                    d = [i,j]  
                    break
                else:
                    continue            
            if d!=[0,0]:
                break
            else:
                continue
        else:
            continue

    for i in range(83):
        for j in range(83):
        
            if image[82-i,j] == 255:
                b = [82-i,j]  
                break
            else:
                continue            
        if b!=[0,0]:
            break
        else:
            continue

    # algorithm for finding radius 2
    #a = [a[0]+190, a[1]+270]
    #b = [b[0]+190, b[1]+270]
    #c = [c[0]+190, c[1]+270]
    #d = [d[0]+190, d[1]+270]
    
    O = [(b[0] + d[0])/2, (a[1] + c[1]) / 2]
    distance = []
    distance.append(np.sqrt((a[0] - O[0]) ** 2 + (a[1] - O[1]) ** 2))
    distance.append(np.sqrt((b[0] - O[0]) ** 2 + (b[1] - O[1]) ** 2))
    distance.append(np.sqrt((c[0] - O[0]) ** 2 + (c[1] - O[1]) ** 2))
    distance.append(np.sqrt((d[0] - O[0]) ** 2 + (d[1] - O[1]) ** 2))

    # setting up appropriate radius
    #if max(distance) < 50:
    R = 0.94*max(distance)+abs((max(O)-50)/8)
    #else:
        #R = 50

    return R

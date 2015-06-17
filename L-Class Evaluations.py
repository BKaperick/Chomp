from chomp import *
from numpy import *

def create_int_matrix(int_list, w = 20, h = 20):
    L2 = matrix([[-1]*w]*h)
    min = int_list[0]-1
    ymin = len(int_list)-1
    for i in range(len(int_list)):
        int_list[i] += 1

    for i in range(min+2, w):
        for j in range(ymin+2, h):
            boardl2 = ChompBoard([i] + int_list+(j-len(int_list)-1)*[1])
            #show(boardl2.matrix())
            num_reachable_l2 = len(find_reachable_P_positions(boardl2))
            L2[i,j]= num_reachable_l2
    return L2

w = 6
h = 24
Output = matrix([[0]*w]*h)
Output[0,0] = 2
#for j in range(1,h):
#            Output[j,i] = j
Output[0,1] = 2
for i in range(2,w):
    if i%3 == 0: 
        Output[0,i] = Output[0,i-3] + 1
        #for j in range(1,h):
        #    Output[j,i] = j
    elif (i-1)%3 == 0: 
        Output[0,i] = 2
#show(Output)
for i in range(0, w, 3):
    print (i)
    Cmatrix = create_int_matrix([Output[0,i], Output[0,i+1]])
    for m in range(20):
        print (m)
        for n in range(20):
            if Cmatrix[m,n] == 0:
                Output[m,i] = m
                Output[m,i+1] = n 

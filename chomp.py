from pickle import *
from copy import copy

class ChompBoard():
    def __init__(self, _rows = []):

        #make sure sequence is non-increasing, then initialize new board
        if all(_rows[i]>=_rows[i+1] for i in range(len(_rows)-1)):
            self._rows = _rows

            #clean up extra zeros at beginning of 1st row.
            while self._rows != [] and self._rows[len(self._rows)-1] == 0:
                self._rows.pop()
        else:
            raise TypeError("list elements must be non-increasing")

    #an equality test, that is used when deciding if a P-board is new or one seen alread
    def __eq__(self, other):
        return self._rows == other._rows

    #repr = "representation", used when printing some representation of a board
    def __repr__(self):
        return str(self._rows)

    #a user-accessible way to get a name for a board
    def name(self):
        return str(self._rows)

    #a non-hidden, user-accessible way to get the list of cookies per row
    def rows(self):
        return copy(self._rows)

    #a non-hidden, user-accessible way to get the list of cookies per column
    def columns(self):
        M = self.matrix()
        return [sum(x) for x in M.columns()]

    #should be the number of rows that have at least one cookie
    def number_of_rows(self):
        return len(self._rows)

    #a non-hidden, user-accessible way to get the list of cookies per column
    def columns(self):
        M = self.matrix()
        return [sum(x) for x in M.columns()]

    #should be the number of cookies in the first row
    def number_of_columns(self):
        if self.number_of_rows() == 0:
            return 0
        else:
            return self._rows[0]

    #counts number of cookies
    def number_of_cookies(self):
        return sum(self._rows)

    #counts full board size
    def full_rectangle(self):
        return self.number_of_columns() * self.number_of_rows()

    #counts empty spaces
    def empty_space(self):
        return self.full_rectangle() - self.number_of_cookies()

    #counts full square
    def full_square(self):
        return max(self.number_of_rows()**2,self.number_of_columns()**2)

    def rows_of_different_length(self):
        M = self.matrix()
        return M.rank()

    # column rank of matrix = row rank
    def columns_of_different_length(self):
        return self.rows_of_different_length()

    # number of duplicate rows
    def duplicate_rows(self):
        return self.number_of_rows() - self.rows_of_different_length()

    # number of duplicate columns
    def duplicate_columns(self):
        return self.number_of_columns() - self.columns_of_different_length()

    # full square matrix
    def squared_matrix(self):
        M = self.matrix()
        M = self.matrix()
        if self._rows == []:
            return matrix(0,0,[])
        #ws[0],len(self._rows))
        M = matrix(m,m,[0]*(m*m))
        for i in range(len(self._rows)):
            for j in range(self._rows[i]):
                M[i,j] = 1
        return M

    # determinant
    def determinant(self):
        M = self.squared_matrix()
        return det(M)

    # trace
    def trace_square(self):
        M = self.squared_matrix()
        return M.trace()

    # rows-columns difference
    def squareness(self):
        return abs(self.number_of_rows()-self.number_of_columns())

    #gives a "picture" of the board as a 0-1 matrix
    def matrix(self):
        if self._rows == []:
            return matrix(0,0,[])
        m = len(self._rows)
        n = self._rows[0]
        M = matrix(m,n,[0]*(m*n))
        for i in range(m):
            for j in range(self._rows[i]):
                M[i,j] = 1
        return M

    #column rank of matrix = row rank
    def columns_of_different_length(self):
        return self.rows_of_different_length()

    # number of duplicate rows
    def duplicate_rows(self):
        return self.number_of_rows() - self.rows_of_different_length()

    # number of duplicate columns
    def duplicate_columns(self):
        n = self._rows[0]
        m = len(self._rows)
        M = matrix(m,n,[0]*(m*n))
        for i in range(m):
            for j in range(self._rows[i]):
                M[i,j] = 1
        return M

    #column rank of matrix = row rank
    def columns_of_different_length(self):
        return self.rows_of_different_length()

    # number of duplicate rows
    def duplicate_rows(self):
        return self.number_of_rows() - self.rows_of_different_length()

    # number of duplicate columns
    def duplicate_columns(self):
        n = self._rows[0]
        M = matrix(m,n,[0]*(m*n))
        for i in range(m):
            for j in range(self._rows[i]):
                M[i,j] = 1
        return M

    #gives min positive eigenvalue
    #DON'T USE, has uninitialized Lreal
    def eigen_max_positive(self):
        M = self.squared_matrix()
        L = M.eigenvalues()
        if Lreal != []:
            return max(Lreal)
        else:
            return 0

    def eigen_min_positive(self):
        M = self.squared_matrix()
        L = M.eigenvalues()
        Lreal = [real_part(x) for x in L if real_part(x) > 0]
        if Lreal != []:
            return max(Lreal)
        else:
            return 0

    #returns a matrix of damage values
    def damage_matrix(self):
        M_columns = self.number_of_columns()
        M_rows = self.number_of_rows()
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M = matrix(M_rows,M_columns,[0]*(M_rows*M_columns))
        oldcookies = self.number_of_cookies()
        for i in range(M_rows):
            for j in range(M_columns):
                newboard= chomp_board_after_eating_cookie(self,i,j)
                newcookies = newboard.number_of_cookies()
                damage = oldcookies - newcookies
                M[i,j] = damage
        return M

      #returns a squared damage matrix
    def squared_damage_matrix(self):
        M_columns = self.number_of_columns()
        M_rows = self.number_of_rows()
        M_max = max(self.number_of_columns(),self.number_of_rows())
        M = matrix(M_max,M_max,[0]*(M_max**2))
        oldcookies = self.number_of_cookies()
        for i in range(M_rows):
            for j in range(M_columns):
                newboard= chomp_board_after_eating_cookie(self,i,j)
                newcookies = newboard.number_of_cookies()
                damage = oldcookies - newcookies
                M[i,j] = damage
        return M

    #returns real eigenvalues
    def eigen(self):
        M = self.squared_matrix()
        L = M.eigenvalues()
        return [x for x in L if x.is_real()]

    #returns a matrix of damage values
    def damage_matrix(self):
        M_columns = self.number_of_columns()
        M_rows = self.number_of_rows()
        number = 0
        M= self.damage_matrix()
        for i in range(M_rows):
            for j in range(M_columns):
                number = number + M[i,j]
        return number

    #returns the gcd of of row lengths
    def row_gcd(self):
        return gcd(self.rows())

    #returns the lcm of of row lengths
    def row_lcm(self):
        return lcm(self.rows())

    #Sum of orthogonal neighbors of each cookie
    def laura_number_sum(self):
        #Trivial cases
        if self._rows == []:
            return 0
        if self.number_of_rows() == 1:
            return 2*self._rows[0] - 2

        #Non-trivial cases
        row1_laura_number = 2*self._rows[0] - 2 + self._rows[1]
        other_row_laura_number = 0

        for i in range(1, self.number_of_rows() - 1):
            other_row_laura_number += 3*self._rows[i] + self._rows[i+1] - 2
        other_row_laura_number += 3*self._rows[-1] - 2

        return row1_laura_number + other_row_laura_number

    #Damage of a cookie is defined as the number of cookies removed from the board if this cookie were to be eaten next move.
    #This is the sum of the damage of each of the cookies on the board.
    def damage_sum(self):
        M_columns = self.number_of_columns()
        M_rows = self.number_of_rows()
        number = 0
        M= self.damage_matrix()
        for i in range(M_rows):
            for j in range(M_columns):
                number = number + M[i,j]
        return number

    #returns the product of the row lengths
    def row_product(self):
        if self.rows() == []:
            return 0
        return prod(self.rows())

    #returns the product of the column lengths
    def column_product(self):
        if self.name() == "[]":
            return 0
        return prod(self.columns())

    #returns the gcd of of column lengths
    def column_gcd(self):
        output = self.column_product()
        for column in self.columns():
            output = gcd(output,column)
        return output

    #returns the lcm of of column lengths
    def column_lcm(self):
        return lcm([sum(x) for x in self.columns()])


    #returns the lcm of of row lengths
    def row_lcm(self):
        return lcm([sum(x) for x in self.rows()])


    # trace of damage matrix
    def trace_damage(self):
        M = self.squared_damage_matrix()
        return M.trace()

    def average_damage(self):
        damage_sum = self.damage_sum()
        cookies = self.number_of_cookies()
        average = damage_sum/cookies
        return average

    def rank_ratio(self):
        rank = self.rows_of_different_length()
        rows = self.number_of_rows()
        return n(rank/rows)



    # gives average number of cookies per row
    def average_cookies_per_row(self):
        count = 0
        rows = self.number_of_rows()
        for cookies in self.rows():
            count = count + cookies
        return n(count/rows)

    # gives average number of cookies per column
    def average_cookies_per_column(self):
        count = 0
        columns = self.number_of_columns()
        for cookies in self.columns():
            count = count + cookies
        return n(count/columns)

    #Steepness of a Chomp Board
    def bevel(self):
        total = 0
        row_diff = 0
        i=0
        while i <= self.number_of_rows()-2:
            if self.rows()[i] - self.rows()[i+1] == 0:
                total = total + row_diff
            else:
                row_diff = self.rows()[i] - self.rows()[i+1]
                total = total + row_diff
            i=i+1
        return total

    #Smallest row size
    def smallest_row_size(self):
        return min(self.rows())

    #Smallest column size
    def smallest_column_size(self):
        M = self.matrix()
        return min(M.columns())

    #number of cookies not in the first row or first column
    def cookies_inside(self):
        cookies = self.number_of_cookies()
        M = self.number_of_rows()
        N = self.number_of_columns()
        if M == 1 or N == 1:
            inside = 0
        else:
            inside = cookies - (M+N-1)
        return inside

    #ratio of cookies not in the first row or first column and total cookies
    def cookies_inside_ratio(self):
        cookies = self.number_of_cookies()
        inside = self.cookies_inside()
        ratio = n(inside/cookies)
        return ratio

    # remainder when cookies is divided by rows
    def row_remainder(self):
        M = self.number_of_rows()
        cookies = self.number_of_cookies()
        remainder = mod(cookies,M)
        return remainder

    # remainder when cookies is divided by columns
    def column_remainder(self):
        N = self.number_of_columns()
        cookies = self.number_of_cookies()
        remainder = mod(cookies,N)
        return remainder

    #For test_completes_self
    def row_difference(board):
        M = []
        for x in range(len(board)):
            if x < len(board)-1:
                M.append(board[x]-board[x+1])
        M.append(board[len(board)-1])
        return M

    #For test_completes_self
    def column_difference(board):
        M = []
        N = []
        T = ChompBoard([5,4,2,1]).matrix().transpose()
        for x in range(board[0]):
            M.append(sum(T[x]))
        return filter(lambda x: x != 0, row_difference(M))

    #For test_completes_self
    def step_list(board):
        A = row_difference(board)
        B = column_difference(board)
        C = []
        for x in range(len(A)):
            C.append(B[x])
            C.append(A[x])
        return C

        ###### PROPERTIES

    #true if odd number of cookies false if even
    def is_P_board(self):
        if is_P_position(self):
            return True
        else:
            return False

    def is_N_board(self):
        if is_N_position(self):
            return True
        else:
            return False

    def odd_cookies(self):
        number = self.number_of_cookies()
        if is_even(number):
            return False
        else:
            return True

    #true if even number of cookies false if odd
    def even_cookies(self):
        number = self.number_of_cookies()
        if is_even(number):
            return True
        else:
            return False

    #tests if the board is a full square board
    def is_square(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        C = self.number_of_cookies()
        if M != N:
            return False
        elif C != M*N:
            return False
        else:
            return True
    def is_rectangle(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        C = self.number_of_cookies()
        if C != M*N:
            return False
        else:
            return True


    #test if our key conjecture is true (number_of_cookes >= 2*number_of_columns - 1)

    def test_theorem1(self):
        C = self.number_of_cookies()
        col = self.number_of_columns()
        if C >= 2*col - 1:
            return True
        else:
            return False

    # tests if the squareness == 0 (rows == columns)
    def squareness_is_zero(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        if M == N:
            return True
        else:
            return False

    # checks if the trace is even
    def even_trace(self):
        T = self.trace_square()
        if is_even(T):
            return True
        else:
            return False

    # checks if the trace if odd
    def odd_trace(self):
        T = self.trace_square()
        if is_odd(T):
            return True
        else:
            return False


    #test if our key conjecture is true (number_of_cookes >= 2*number_of_columns - 1)

    def test_columns_columns_theorem(self):
        C = self.number_of_cookies()
        col = self.number_of_columns()
        if C >= 2*col - 1:
            return True
        else:
            return False

    # tests if the board is a balanced L
    def is_balanced_L(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        C = self.number_of_cookies()
        if M == N and C == M + N - 1:
            return True
        else:
            return False

    # tests if the board is an unbalanced L
    def is_unbalanced_L(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        C = self.number_of_cookies()
        if C > M+N-1:
            return False
        elif M == N:
            return False
        elif M == 1 or N == 1:
            return False
        else:
            return True

    #the two column/row solution
    def is_two_row_solution(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        C = self.number_of_cookies()
        if M != 2 and N != 2:
            return False
        elif C != (2*M)-1 and C != (2*N)-1:
            return False
        else:
            return True

    # tests if there are not rows of different
    def rank_one_property(self):
        R = self.rows_of_different_length()
        if R > 1:
            return False
        else:
            return True

    # tests if there are rows of different length
    def rank_non_one_property(self):
        R = self.rows_of_different_length()
        if R > 1:
            return True
        else:
            return False

    #test if our key conjecture is true (number_of_cookes >= 2*number_of_columns - 1)

    def test_theorem1(self):
        C = self.number_of_cookies()
        col = self.number_of_columns()
        if C >= 2*col - 1:
            return True
        else:
            return False

    # tests if a balanced L is reachable
    def reachable_balanced_L(self):
        reachable = find_reachable_boards(self)
        for board in reachable:
            if board.balanced_L() == True:
                return True
        return False

    # if the two row solution is reachable
    # tests if a unbalanced L is reachable
    def reachable_unbalanced_L(self):
        reachable = find_reachable_boards(self)
        for board in reachable:
            if board.unbalanced_L() == True:
                return True
        return False


    # property of p position
    # if the two row solution is reachable
    def reachable_two_row_solution(self):
        reachable = find_reachable_boards(self)
        for board in reachable:
            if board.two_row_solution() == True:
                return True
        return False

    # property of p position
    def test_P_position(self):
        if is_P_position(self)== True:
            return True
        else:
            return False

    #zeilberger paper theorem 1 about 3 row chomp
    def three_row_one_cookie(self):
        M_rows = self.number_of_rows()
        if M_rows == 3:
            if self._rows[2]==1:
                if self.rows[0] == 2 and self.rows[1]==2:
                    return True
                elif self.rows[1] == 1 and self.rows[0]==3:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def is_almost_L(self):
        M_rows = self.number_of_rows()
        M_col = self.number_of_columns()
        cookies = self.number_of_cookies()
        if abs(M_rows-M_col) != 1:
            return False
        elif cookies != M_rows + M_col:
            return True
        else:
            return False

    def test_prime_cookies(self):
        cookies = self.number_of_cookies()
        if is_prime(cookies):
            return True
        else:
            False

    def test_row_theorem(self):
        M = self.number_of_rows()
        cookies = self.number_of_cookies()
        if cookies >= 2*M - 1:
            return False
        elif is_even(M_rows) and M_rows < M_col:
            return True
        elif is_even(M_col) and M_col < M_rows:
            return True
        else:
            return False


    #zeilberger paper theorem 2 about 3 row chomp
    def three_row_two_cookie(self):
        M_rows = self.number_of_rows()
        if M_rows == 3:
            if self._rows[2]==2 and (self.rows[0]-self.rows[1])==2:
                return True
            else:
                return False
        else:
            return False

    #needs explanation
    def is_almost_L(self):
        M_rows = self.number_of_rows()
        M_col = self.number_of_columns()
        cookies = self.number_of_cookies()
        if abs(M_rows-M_col) != 1:
            return False
        elif cookies != M_rows + M_col:
            return True
        else:
            return False

    def test_prime_cookies(self):
        cookies = self.number_of_cookies()
        if is_prime(cookies):
            return True
        else:
            False

    def test_row_theorem(self):
        M = self.number_of_rows()
        cookies = self.number_of_cookies()
        if cookies >= 2*M - 1:
            return False
        elif is_even(M_rows) and M_rows < M_col:
            return True
        elif is_even(M_col) and M_col < M_rows:
            return True
        else:
            return False


    def test_Theorem2(self):
        M_rows = self.number_of_rows()
        M_col = self.number_of_columns()
        cookies = self.number_of_cookies
        if M_col == M_rows + 1 and cookies == M_rows + M_col:
            return True
        else:
            return False

    def test_prime_cookies(self):
        cookies = self.number_of_cookies()
        if is_prime(cookies):
            return True
        else:
            False

    def test_row_theorem(self):
        M = self.number_of_rows()
        cookies = self.number_of_cookies()
        if cookies >= 2*M - 1:
            return True
        else:
            return False
    def composite_odd_cookies(self):
        cookies = self.number_of_cookies()
        if is_even(cookies):
            return False
        elif is_prime(cookies):
            return False
        else:
            return True

    def rows_greater_than_columns(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        if M > N :
            return True
        else:
            return False

    def columns_greater_than_rows(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        if M < N :
            return True
        else:
            return False

    def is_symmetric(self):
        board = self.rows()
        transpose = self.columns()
        if board == transpose:
            return True
        else:
            return False

    def cookies_in_greater_than_cookies_out(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        cookies = self.number_of_cookies()
        if M == 1 or N == 1:
            return False
        elif cookies-(M+N-1) > M+N-1:
            return True
        else:
            return False
    def cookies_out_greater_than_cookies_in(self):
        M = self.number_of_rows()
        N = self.number_of_columns()
        cookies = self.number_of_cookies()
        if M == 1 or N == 1:
            return True
        elif cookies-(M+N-1) < M+N-1:
            return True
        else:
            return False

    def cookies_divisible_by_rows(self):
        M = self.number_of_rows()
        cookies = self.number_of_cookies()
        if mod(cookies,M) == 0:
            return True
        else:
            return False

    def cookies_divisible_by_columns(self):
        N = self.number_of_columns()
        cookies = self.number_of_cookies()
        if mod(cookies,N) == 0:
            return True
        else:
            return False



    #If rotated, does chomp board fit nicely with itself?
    def test_completes_self(board):
        A = step_list(board)[0:(len(step_list(board))-1)]
        B = step_list(board)[1:len(step_list(board))]
        if A == A[::-1] or B == B[::-1]:
            return True
        else:
            return False


####FUNCTIONS

def string_to_list(string):
    L = []
    for x in str(string[1:(len(string)-1):3]):
        L.append(int(x))
    return L

#eat cookie in i,j th spot on board and return updated board
def chomp_board_after_eating_cookie(board,i,j):
    board_rows = board.rows()

    m = board.number_of_rows()
    n = board.number_of_columns()

    new = board.rows()

    for k in range(i,m):
        new[k] = min(board_rows[k],j)

    return ChompBoard(new)



#find all the boards that are possible after the current player's move
def find_reachable_boards(board):

    reachable = []

    if board.rows() == []:
        return reachable

    m = board.number_of_rows()
    n = board.number_of_columns()

    for i in range(m):
        for j in range(n):
            new = chomp_board_after_eating_cookie(board,i,j)
            if new.rows() != board.rows():
                if new.number_of_rows() >= 1:
                    reachable.append(new)

    return reachable

#finds the move i,j that was made to go from oldboard to newboard
def find_move(oldboard, newboard):
    row = -1
    column = -1
    if oldboard.number_of_columns() != newboard.number_of_columns():
        return (0, newboard.rows()[0])

    elif oldboard.number_of_rows() != newboard.number_of_rows():
        return (newboard.number_of_rows(), 0)

    else:
        old= oldboard.matrix()
        newmatrix = newboard.matrix()
        for row in range(oldboard.number_of_rows()):
            for cookie in range(oldboard.rows()[row]):
                if oldmatrix[row][cookie] != newmatrix[row][cookie]:
                    return (row, cookie)

# a position B is an P-position iff every reachable position is an N-position
#equivalently, B is an N-position iff not (every reachable position is an N-position)
#equivalently, B is an N-position iff there is a reachable position which is an P-position
# base case: no cookies left is an N-position (and the sum of the rows = 0)

#stores "N" for postions which are N-positions and "P" for positions which are P-positions
#loads "position_values.sobj" if it exists otherwise initializes a new one
try:
    temp = load("position_values.sobj")
    position_values = temp
except:
    position_values = {str([]):"N"}

#tests if a position is a N-position
def is_N_position(B):

    if B.name() in position_values:
        if position_values[B.name()] == "N":
            return True
        else:
            return False

    else:
        value = any(is_P_position(A) for A in find_reachable_boards(B))
        if value == True:
            position_values[B.name()] = "N"
            return True
        else:
            position_values[B.name()] = "P"
            return False

#tests if a position is a P-position
def is_P_position(B):

    if B.name() in position_values:
        if position_values[B.name()] == "P":
            return True
        else:
            return False
    else:
        value = is_N_position(B)
        if value == True:
            position_values[B.name()] = "N"
            return False
        else:
            position_values[B.name()] = "P"
            return True

#finds all the positions that the current player can choose for her next move that are winners
#includes a "save" function - so that position_values is saved occsionally

def find_reachable_P_positions(B):
    reachable = find_reachable_boards(B)
    reachable_P = []

    for x in reachable:
        if is_P_position(x):
            reachable_P.append(x)

    #can be removed
    #save(position_values, "position_values.sobj")

    return reachable_P

#finds all the positions that the current player can choose for her next move that are losers
def find_reachable_N_positions(B):
    reachable = find_reachable_boards(B)
    reachable_N = []

    for x in reachable:
        if not is_P_position(x):
            reachable_N.append(x)

    return reachable_N


#saves the current position_values dictionary into a file "known_positions.p" in the main chomp directory.
#It will overwrite whatever is currently in the "known_positions.p" file.
def save_position_values():
    output = open('known_positions.p', 'wb')
    pickle.dump(position_values, output)
    output.close()

#stores the contents of "known_position.p" file in the  position_values dictionary.
def load_position_values():
    global position_values
    input = open('known_positions.p', 'rb')
    position_values = pickle.load(input)


#Prints out the positions in position_values dictionary in the following format:
#P Positions:
#ChompBoard([r1,r2,...,rn]), ChompBoard([r1,r2,...,rn]),...
#
#
#N Positions:
#ChompBoard([r1,r2,...,rn]), ChompBoard([r1,r2,...,rn]),...
#
#
#A word of caution: sage seems to cut off strings that are too long, so when dealing with a lot of board positions, it may end the string with "[...]".  Just remove this and any partial ChompBoard declaration and you should be good.
#Also: the conjecturing program has a maximum object limit so you may need to cut out some of the objects when trying to conjecture with them.
def print_position_values_formatted():
    p_out = "P-Positions:\n"
    n_out = "N-Positions:\n"
    for k, v in position_values.iteritems():
        if v == 'P':
            p_out += "ChompBoard({}), ".format(k)
        elif v == 'N':
            n_out += "ChompBoard({}), ".format(k)
    print (p_out,'\n\n', n_out)

#Print P-positions only
def print_P_position_values_formatted():
    p_out = "P-Positions:\n"
    n_out = "N-Positions:\n"
    for k, v in position_values.iteritems():
        if v == 'P':
            p_out += "ChompBoard({}), ".format(k)
        elif v == 'N':
            n_out += "ChompBoard({}), ".format(k)
    print (p_out)

#Print N-positions only
def print_N_position_values_formatted():
    p_out = "P-Positions:\n"
    n_out = "N-Positions:\n"
    for k, v in position_values.iteritems():
        if v == 'P':
            p_out += "ChompBoard({}), ".format(k)
        elif v == 'N':
            n_out += "ChompBoard({}), ".format(k)
    print (n_out)

import ast
def convert_board_name_to_board(name_of_board):
    L = ast.literal_eval(name_of_board)
    return ChompBoard(L)

#Pexamples is a list of names of previously computed P positions
Pexamples = [board_name for board_name in position_values if position_values[board_name]=="P"]

#Nexamples is a list of names of previously computed N positions
Nexamples = [board_name for board_name in position_values if position_values[board_name]=="N"]


#Pcounterexamples is a list of names of P-boards that have been counterexamples to conjectures
#initializes to list with only poison-cookie board if load fails

try:
    temp = load("Pcounterexamples.sobj")
    Pcounterexamples = temp
except:
    Pcounterexamples = ["[1]"]

try:
    temp = load("Ncounterexamples.sobj")
    Ncounterexamples = temp
except:
    Ncounterexamples = ["[2]"]

#find a counterexample P-board to a *single* conjecture if one exists among all boards of size no more than
#row_limit x column_limit
#first it checks if any example in Pcounterexamples is already known
#should *not* add duplicates into Pcounterexamples list

def find_Pcounterexample(conj, row_limit, column_limit):

    #tests existing P-counterexamples first
    for s in Pcounterexamples:
        board = convert_board_name_to_board(s)
        try:
            value = conj.evaluate(board)
            if value == False:
                #Pcounterexamples.append(board.name())
                print ("{} is a counterexample to: {}".format(board.name(),conj))
                return board
        except:
            print ("error with evaluating: {} for {}".format(conj,board.name()))

    L = find_reachable_P_positions(ChompBoard([column_limit]*row_limit))

    for board in L:
        try:
            value = conj.evaluate(board)
            if value == False and board.name() not in Pcounterexamples:
                Pcounterexamples.append(board.name())
                print ("{} is a counterexample to: {}".format(board.name(),conj))
                return board
        except:
            print ("error with evaluating: {} for {}".format(conj,board.name()))

    print ("{}: No counterexamples found".format(conj))


def find_Ncounterexample(conj, row_limit, column_limit):

    #tests existing N-counterexamples first
    for s in Ncounterexamples:
        board = convert_board_name_to_board(s)
        try:
            value = conj.evaluate(board)
            if value == False:
                print ("{} is a counterexample to: {}".format(board.name(),conj))
                return board
        except:
            print ("error with evaluating: {} for {}".format(conj,board.name()))

    L = find_reachable_N_positions(ChompBoard([column_limit]*row_limit))

    for board in L:
        try:
            value = conj.evaluate(board)
            if value == False and board.name() not in Ncounterexamples:
                Ncounterexamples.append(board.name())
                print ("{} is a counterexample to: {}".format(board.name(),conj))
                return board
        except:
            print ("error with evaluating: {} for {}".format(conj,board.name()))

    print ("{}: No counterexamples found".format(conj))

#finds Pcounterexamples to a *list* of P-board conjectures
#saves Pcounterexample file
# inefficient: scans position_values and filters Ppositions multiple times - could be written to do this once

def find_Pcounterexamples(conjs, row_limit, column_limit):

    for conj in conjs:
        find_Pcounterexample(conj, row_limit, column_limit)

    #save(Pcounterexamples, "Pcounterexamples.sobj")

def find_Ncounterexamples(conjs, row_limit, column_limit):

    for conj in conjs:
        find_Ncounterexample(conj, row_limit, column_limit)

    #save(Ncounterexamples, "Ncounterexamples.sobj")


# appends conjectures with n counterexample to new list
def find_p_unique_conjs(conjs, row_limit, column_limit):
    pconjs= []
    for i in range(len(conjs)): ##Bryan changed this line at 1:13 AM on 5/23/15.  Previously it was "for i in [0..len(conjs)-1]:"  if I missed some nuance when I changed it, please correct it, but the previous was not compiling.
        if find_Ncounterexample(conjs[i],row_limit,column_limit) != []:
            pconjs.append(conjs[i])
    return pconjs


## SPECIAL OBJECTS

#UCLA's board
ucla = ChompBoard([7,7,7,7])

Nspecial = ["[2]", "[10]", "[3,2,1]", "[4,2]", "[7,2]", "[7,7,7,7]", "[3,2,1,1]", "[3,3,3,3,3,3,3,3,3,3,2]", "[24,13,13,13,13,6]", "[28,25,15]", "[15,15,9,6]", "[2,2,2,2,2,2,2,2,2,2,2,2]"] + Ncounterexamples

Pspecial = ["[2,1]", "[1]", "[5,1,1,1,1]", "[3,2]", "[7,6]", "[100,45,45,45]", "[3,3,3,3,3,3,1,1,1,1,1]", "[24,13,7,7,7,6]", "[28,17,15]", "[15,13,9,6]", "[2,2,2,2,2,2,2,2,2,2,2,1]", "[11,11,10,9,8,8,5,5]"] + Pcounterexamples



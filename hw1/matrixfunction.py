def transpose(M):
    row = len(M)
    try:
        col = len(M[0])
    except TypeError:
        col = row
        row = 1
        M = [M]
    M_T = [[0]*row for _ in range(col)]
    for i in range(row):
        for j in range(col):
            M_T[j][i] = M[i][j]
    return M_T


def mul(A, B):
    row_A = len(A)
    row_B = len(B)
    col_B = len(B[0])

    M = [[0]*col_B for _ in range(row_A)]
    for x in range(row_A):
        for i in range(col_B):
            sum = 0
            for j in range(row_B):
                sum += A[x][j] * B[j][i]
            M[x][i] = sum
    return M

def LU_decomposition(M):
    row = len(M)
    U = M.copy()
    L = [[1]*row for _ in range(row)]
    for i in range(row):
        for j in range(i+1, row):
            L[j][i] = U[j][i] / U[i][i]
            L[i][j] = 0
            U[j] = [U[j][x] - L[j][i] * U[i][x] for x in range(row)]   
    return L, U

def inverse(M):
    L, U = LU_decomposition(M)
    row = len(M)
    M_inverse = []
    
    '''
    LUA^-1 = I, Y = UA^-1, LY = I
    find Y : Y = UA^-1
    '''

    #LY=I
    Temp = []
    for k in range(row):
        x = [0] * row
        x[k] = 1
        for i in range(k+1, row):
            for j in range(k,i):
                x[i] += L[i][j] * x[j]
            x[i] *= -1
        Temp.append(x)        
    
    #Y=UA^-1
    for k in range(row):
        x = Temp[k].copy()

        for i in range(row-1, -1, -1 ):
            for j in range(row-1, i, -1):
                x[i] = x[i] - U[i][j] * x[j]
            x[i] = x[i] / U[i][i]
        M_inverse.append(x)
    return(transpose(M_inverse))
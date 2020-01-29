from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

print('Введите количество точек по оси z')
n_z = int(input())

print('Введите количество точек по оси rho')
n_rho = int(input())

z_0, z_f = 0, 200
rho_0, rho_f = 300, 320

points = []
rho = np.linspace(rho_0, rho_f, n_rho)
z = np.linspace(z_0, z_f, n_z)
points = []
for r_i in rho:
    for z_i in z:
        points.append([z_i, r_i])
points = np.array(points)
Th = Delaunay(points) #Делаем триангуляцию Делоне

'''
Везде далее: 
K - координаты симплексного КЭ в R^2
K = [[z1,r1],[z2,r2],[z3,r3]]
'''

 #Вычиление Якобиана
def detB(K):
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    return abs((r1-r3)*(z2-z3) - (r2-r3)*(z1-z3))


def G_k(K, lamda_coef=135): #Матрица локальной СЛАУ по K
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    
    det_B = detB(K)
    coef = det_B * 1/6 * (r1+r2+r3)
    
    A = np.array([[1,r1,z1],[1,r2,z2],[1,r3,z3]])
    a1,b1,c1 = np.linalg.solve(A, [1,0,0])
    a2,b2,c2 = np.linalg.solve(A, [0,1,0])
    a3,b3,c3 = np.linalg.solve(A, [0,0,1])
    
    B_k = np.array([[b1,b2,b3],[c1,c2,c3]])
    B_k_T = B_k.T
    lambda_mat = np.array([[lamda_coef,0],[0,lamda_coef]])
    
    G_k = coef * np.matmul(np.matmul(B_k_T, lambda_mat), B_k)
    return G_k  

def G_SK_1(K, alpha_T=10): #Добавка к локальной матрице по границе КЭ (F_1)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    
    J = np.linalg.norm(K[1]-K[2])
    coef = alpha_T * 1/12 * J
    
    matrix = np.array([[0,0,0],[0,3*r2+r3,r2+r3],[0,r2+r3,r2+3*r3]])
    return -coef*matrix 

def G_SK_2(K, alpha_T=10): #Добавка к локальной матрице по границе КЭ (F_2)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    
    J = np.linalg.norm(K[0]-K[2])
    coef = alpha_T * 1/12 * J
    
    matrix = np.array([[3*r1+r3,0,r1+r3],[0,0,0],[r1+r3,0,r1+3*r3]])
    return -coef*matrix   

def G_SK_3(K, alpha_T=10): #Добавка к локальной матрице по границе КЭ (F_3)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    
    J = np.linalg.norm(K[0]-K[1])
    coef = alpha_T * 1/12 * J
    
    matrix = np.array([[3*r1+r2,r1+r2,0],[r1+r2,r1+3*r2,0],[0,0,0]])
    return -coef*matrix  


def f_k(K, f=0): #Вектор правой части по КЭ
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    
    det_B = detB(K)
    Tqe = np.array([[f,f,f]])
    
    matrix = np.array([[3*r1+r2+r3,r1+r2+1/2*r3,r1+1/2*r2+r3],
                       [r1+r2+1/2*r3,r1+3*r2+r3,1/2*r1+r2+r3],
                       [r1+1/2*r2+r3,1/2*r1+r2+r3,r1+r2+3*r3]])

    f = det_B / 60 * np.matmul(Tqe, matrix)
    return f  



def f_SK_1(K, q_e=30, alpha_T=10, theta_inf = 273.15): #Добавка к вектору правой части по границе КЭ (F_1)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    Tqe = np.array([[q_e,q_e,q_e]])
    
    J23 = np.linalg.norm(K[1]-K[2])
    coef = 1/12 * J23
    matrix = np.array([[0,0,0],[0,3*r2+r3,r2+r3],[0,r2+r3,r2+3*r3]])

    
    I1 = alpha_T * theta_inf * 1/6 * J23 * np.array([[0, 2*r2+r3, r2+2*r3]])
    
    return -coef*np.matmul(Tqe, matrix)-I1

def f_SK_2(K, q_e=30, alpha_T=10, theta_inf = 273.15): #Добавка к вектору правой части по границе КЭ (F_2)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    Tqe = np.array([[q_e,q_e,q_e]])
    
    J13 = np.linalg.norm(K[0]-K[2])
    coef = 1/12 * J13
    matrix = np.array([[3*r1+r3,0,r1+r3],[0,0,0],[r1+r3,0,r1+3*r3]])
    
    I2 = alpha_T * theta_inf * 1/6 * J13 * np.array([[2*r1+r3, 0, r1+2*r3]])
    
    return -coef*np.matmul(Tqe, matrix)-I2

def f_SK_3(K, q_e=30, alpha_T=10, theta_inf = 273.15): #Добавка к вектору правой части по границе КЭ (F_3)
    z1 = K[0][0]
    z2 = K[1][0]
    z3 = K[2][0]
    r1 = K[0][1]
    r2 = K[1][1]
    r3 = K[2][1]
    Tqe = np.array([[q_e,q_e,q_e]])
    
    J12 = np.linalg.norm(K[0]-K[1])
    coef = 1/12 * J12
    matrix = np.array([[3*r1+r2,r1+r2,0],[r1+r2,r1+3*r2,0],[0,0,0]])
    
    I3 = alpha_T * theta_inf * 1/6 * J12 * np.array([[2*r1+r2, 2*r2+r1, 0]])
    
    return -coef*np.matmul(Tqe, matrix)-I3


print('...Сборка СЛАУ...')

Sigma_q = [] #триангуляция поверхности, на которой ставится граничное условие третьего рода
for K in Th.simplices:
    if points[K[1]][1] == 300 and points[K[2]][1] == 300:
        Sigma_q.append([K, 1])
    if points[K[0]][1] == 300 and points[K[2]][1] == 300:
        Sigma_q.append([K, 2])
    if points[K[1]][1] == 300 and points[K[0]][1] == 300:
        Sigma_q.append([K, 3])
        
    if points[K[1]][1] == 320 and points[K[2]][1] == 320:
        Sigma_q.append([K, 1])
    if points[K[0]][1] == 320 and points[K[2]][1] == 320:
        Sigma_q.append([K, 2])
    if points[K[1]][1] == 320 and points[K[0]][1] == 320:
        Sigma_q.append([K, 3])
        
Sigma_theta = [] #множество индексов узлов, принадлежащих поверхности, на которой ставится граничное условие первого рода
for i, p in enumerate(points):
    if p[0] == 0 or p[0] == 200:
        Sigma_theta.append(i)

theta_e = 298.15

f = np.zeros(np.max(Th.simplices)+1)
G = np.zeros((np.max(Th.simplices)+1, np.max(Th.simplices)+1))


#сборка правой части глобальной СЛАУ
for K in Th.simplices:
    for i in range(len(K)):
        f[K[i]] += f_k(points[K])[0][i]

#учет ГУ третьего рода
for K, ind in Sigma_q:
    S_k = K
    for i in range(3):
        for j in range(3):
            if ind == 1:
                G[S_k[i]][S_k[j]] += G_SK_1(points[K])[i,j]
            if ind == 2:
                G[S_k[i]][S_k[j]] += G_SK_2(points[K])[i,j]
            if ind == 3:
                G[S_k[i]][S_k[j]] += G_SK_3(points[K])[i,j]
        if ind == 1:
            f[S_k[i]] += f_SK_1(points[K])[0][i]
        if ind == 2:
            f[S_k[i]] += f_SK_2(points[K])[0][i]
        if ind == 3:
            f[S_k[i]] += f_SK_3(points[K])[0][i]

#учет ГУ первого рода
for i in Sigma_theta:
    f[i] = theta_e
    G[i] = 0
    G[i][i] = 1

#Сборка матрицы глобальной СЛАУ
for K in Th.simplices:
    for i in range(len(K)):
        if K[i] not in Sigma_theta:
            for j in range(len(K)):
                if K[j] not in Sigma_theta:
                    G[K[i]][K[j]] += G_k(points[K])[i][j]
                else:
                    f[K[i]] -= G_k(points[K])[i][j]*f[K[j]]



print('...Решение СЛАУ...')
#решение глобальной СЛАУ
theta = np.linalg.solve(G, f)

from scipy.interpolate import interp2d, NearestNDInterpolator, LinearNDInterpolator
interp_t = NearestNDInterpolator(points, theta)
Xm, Ym = np.meshgrid(z, rho)
theta_grid = interp_t(Xm, Ym)


#запись в файл в формате .mv2 
with open('result.mv2', 'w') as f:
    f.write(str(len(points))+ ' '+str(3)+' '+str(1)+' theta \n')
    for i, p in enumerate(points):
        f.write(str(i+1)+ ' '+str(p[1])+ ' '+ str(p[0])+ ' 0'+' '+str(theta[i])+' \n')
    f.write(str(len(Th.simplices))+ ' '+str(3)+' '+str(3)+' BC_id mat_id mat_id_Out \n')
    for i, K in enumerate(Th.simplices):
        f.write(str(i+1)+ ' '+str(K[0]+1)+' '+str(K[1]+1)+' '+str(K[2]+1)+' 1 1 1 0 \n')

plt.figure(dpi = 200)
plt.xlim(-10, 210)
plt.ylim(290, 330)
plt.pcolor(Xm, Ym, theta_grid)
plt.colorbar()
plt.show()

import numpy as np

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    #TODO
    # begin answer
    from scipy.optimize import minimize
    
    # 构造核矩阵
    K = X.T @ X
    y_flat = y.flatten()
    
    # 定义目标函数
    def objective(alpha):
        quad_term = 0.5 * alpha.T @ ((y_flat[:, None] @ y_flat[None, :]) * K) @ alpha
        linear_term = -np.sum(alpha)
        return quad_term + linear_term
    
    # 定义约束条件
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y_flat)}
    bounds = [(0, None) for _ in range(N)]#规定下界是0 无上界
    
    # 求解
    alpha_init = np.ones(N) / N
    result = minimize(objective, alpha_init, method='SLSQP', bounds=bounds, constraints=constraints)
    alpha = result.x#result.x指的是最优解的自变量值
    
    # 从alpha计算w
    w_weights = np.sum((alpha * y_flat)[:, None] * X.T, axis=0)
    w[1:] = w_weights.reshape(-1, 1)
    
    # 计算支持向量数量和b
    sv_indices = np.where(alpha > 1e-5)[0]
    num = len(sv_indices)
    
    if num > 0:
        # 使用支持向量计算b
        sv_idx = sv_indices[0]
        w[0, 0] = y_flat[sv_idx] - w_weights @ X[:, sv_idx]
    # end answer
    return w, num


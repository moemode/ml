import numpy as np

# Define the matrix with '?' representing unknown values
Y = np.array([
    [5, np.nan, 7],
    [np.nan, 2, np.nan],
    [4, np.nan, np.nan],
    [np.nan, 3, 6]
])
U_0 = np.array([[6, 0, 3, 6]]).T
V_0 = np.array([[4, 2, 1]])
X_0 = np.array([[24,12,6],[0,0,0],[12,6,3], [24,12,6]])
# ensure product of U_0 and V_0 is X_0
assert np.allclose(U_0 @ V_0, X_0)

mask = ~np.isnan(Y)
SE = 0.5*np.sum((Y[mask]-X_0[mask])**2)
print(f"SE: {SE}")

lambd = 1
reg_U = (lambd/2)*(np.sum(U_0**2))
reg_V = (lambd/2)*(np.sum(V_0**2))
reg = reg_U + reg_V
print(f"Regularization term: {reg_U}")
print(f"Regularization term: {reg_V}")
print(f"Regularization term: {reg}")
import numpy as np
import scipy.optimize as optim


def func_logistic(x, c, k, m):
    y = c / (1 + np.exp(-k*(x-m)))
    return y


def func_gaussian(x, a, b, c):
    # Gaussian function: f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)
    y = a * np.exp(-0.5 * ((x-b)/c)**2)
    return y


def fit_curve_gauss(df, column):
    y_values = df[column].values
    #print(f"y_values: {y_values}")

    x_values = np.arange(len(df[column])) + 1
    #print(f"x_values: {x_values}")

    p0 = [np.max(y_values), np.mean(y_values), np.std(y_values)]
    print(f"p0: {p0}")

    (a,b,c),cov = optim.curve_fit(func_gaussian, x_values, y_values, p0=p0, maxfev=100000)
    #print(f'a: {a}, b: {b}, c: {c}')

    fitted = func_gaussian(x_values, a, b, c)
    #print(f'fitted: {fitted}')
    return (a,b,c), fitted


def func_logistic(x, c, k, m):
    y = c / (1 + np.exp(-k*(x-m)))
    return y


def fit_logistic(df, column):
    X = np.arange(len(df[column])) + 1
    print(f"X: {X}")

    Y = df[column].values
    print(f"Y: {Y}")

    p0 = [np.max(Y), 1, X[int(len(X) / 2)]]
    print(f"p0: {p0}")

    (a,b,c),cov = optim.curve_fit(func_logistic, X, Y, p0=p0, maxfev=100000)
    print(f'a: {a}, b: {b}, c: {c}')

    fitted = func_logistic(X, a, b, c)
    return (a,b,c), fitted


def add_increment(df, column):
    column_inc = f"{column}_new"
    df[column_inc] = df[column].diff()
    df[column_inc].replace(np.nan, 0, inplace=True)
    df.loc[df[column_inc] < 0, column_inc] = np.nan
    df[column_inc] = df[column_inc].interpolate()
    return df
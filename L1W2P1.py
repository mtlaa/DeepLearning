
import numpy as np


def sigmoid(x):
    temp = np.exp(-x) + 1
    return 1/temp


# x = np.array([1, 2, 3])
# print(x.shape)
# print(sigmoid(x))
# print(sigmoid(3))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s*(1-s)


x = np.array([1, 2, 3])
print("sigmoid grad = " + str(sigmoid_grad(x)))


def image2vector(image):
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
# print("image to vector : " + str(image2vector(image)))


def normalizerows(x):
    v = np.linalg.norm(x, axis=1, keepdims=True)
    return x/v


x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = \n" + str(normalizerows(x)))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1).reshape(ex.shape[0], 1)
    print(ex.shape)
    print(sum_ex.shape)
    return ex/sum_ex


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = \n" + str(softmax(x)))


def L1_loss(y_hat, y):
    l = np.abs(y-y_hat)
    return np.sum(l)


def L2_loss(y_hat, y):
    l = np.abs(y-y_hat)
    return np.sum(np.dot(l, l))



yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1_loss(yhat, y)))
print("L2 = " + str(L2_loss(yhat, y)))

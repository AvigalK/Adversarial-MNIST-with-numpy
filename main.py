import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

eps = 1e-8
lr = 0.0001


def plot_image(X, title, ax):
    ax.imshow(X, cmap='gray')
    ax.axis('off')
    ax.set_title(title, color='black', fontsize=10)


def softmax(x):
    e_x = np.exp(x - np.max(x))

    return (e_x.T / (e_x.sum(axis=1) + eps)).T


def cross_entropy(y_train, y_pred):
    M = y_train.shape[0]
    P = -np.log(y_pred[range(M), np.argmax(y_train, axis=1)] + eps)
    CE_loss = np.sum(P) / M
    return CE_loss


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def forward_pass(x_batch, w1, b1, w2, b2):
    Z1 = x_batch @ w1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ w2 + b2
    A2 = softmax(Z2)  # y_pred
    return Z1, A1, Z2, A2


def accuracy(y_train, y_pred):
    acc_bool = (np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1))
    acc_int = acc_bool.astype(int)
    acc = np.mean(acc_int)
    return acc


def test(x, y, w1, b1, w2, b2):
    Z1, A1, Z2, A2 = forward_pass(x, w1, b1, w2, b2)
    loss = cross_entropy(y, A2)
    acc = accuracy(y, A2)
    return loss, acc


digits = 10
train_examples = y_train.shape[0]  # num of rows, pictures
val_examples = y_test.shape[0]
y_train = y_train.reshape(1, train_examples)  # lables in one row
y_val = y_test.reshape(1, val_examples)
Ytrain_new = np.eye(digits)[y_train.astype('int32')]
Ytrain_new = Ytrain_new.T.reshape(digits, train_examples)  # labels one hot encoding in columns
Yval_new = np.eye(digits)[y_val.astype('int32')]
Yval_new = Yval_new.T.reshape(digits, val_examples)  # labels one hot encoding in columns
y_train = Ytrain_new.T
y_test = Yval_new.T

# ==============================================================================================
# training the model
# -------------------

batch_size = 32
N = X_train.shape[0]
epochs = 20

inputLayerNeurons = X_train.shape[1]
H = 32
output_layer = 10
w1 = np.random.normal(loc=0.0, scale=0.01, size=(inputLayerNeurons, H))
b1 = np.random.normal(loc=0.0, scale=0.01, size=(1, H))
w2 = np.random.normal(loc=0.0, scale=0.01, size=(H, output_layer))
b2 = np.random.normal(loc=0.0, scale=0.01, size=(1, output_layer))

train_losses, train_accuracy, val_losses, val_accuracy = ([] for i in range(4))

for epoch in range(epochs):
    loss = 0
    batch_accuracy = []

    for batch_idx, idx_start in enumerate(range(0, N, batch_size)):
        idx_end = min(idx_start + batch_size, N)
        x_batch = X_train[idx_start:idx_end, :]  # take all data in the current batch
        y_batch = y_train[idx_start:idx_end]  # take relevant labels

        # forward pass, notice A2 is predicted prob
        Z1, A1, Z2, A2 = forward_pass(x_batch, w1, b1, w2, b2)

        # batch accuracy list update
        batch_acc = accuracy(y_batch, A2)
        batch_accuracy.append(batch_acc)
        batch_loss = cross_entropy(y_batch, A2)
        loss += batch_loss

        # BP
        # dZ2 = A2 - y_batch
        # dw2 = (1. / batch_size) * np.matmul(A1.T, dZ2)
        # db2 = (1. / batch_size) * np.sum(dZ2, axis=0, keepdims=True)
        # dA1 = np.matmul(dZ2, w2.T)
        # dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        # dw1 = (1. / batch_size) * np.matmul(x_batch.T, dZ1)
        # db1 = (1. / batch_size) * np.sum(dZ1, axis=0, keepdims=True)

        dZ2 = A2 - y_batch
        dw2 = np.matmul(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.matmul(dZ2, w2.T)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dw1 = np.matmul(x_batch.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        w2 -= lr * dw2
        b2 -= db2 * lr
        w1 -= lr * dw1
        b1 -= db1 * lr

    # VALIDATION
    # ------------
    # run model on validation examples
    # val_loss, val_acc = test(x_val,y_val,w1,b1,w2,b2)

    # train accuracy calc
    train_acc = np.mean(batch_accuracy)

    # save for plotting
    train_losses.append(loss / batch_idx)
    train_accuracy.append(train_acc)
    # val_losses.append(val_loss)
    # val_accuracy.append(val_acc)

    # print('loss', loss / batch_idx, ' loss_val', val_loss) # each epoch
    print('loss', loss / batch_idx)

# print('last train_losses:', train_losses[-1], ' last train_accuracy:', train_accuracy[-1], ' last val_losses:', val_losses[-1],' last val_accuracy:', val_accuracy[-1])


# # create plot template
# fig , axes = plt.subplots(8,8, figsize=(24,10))
# fig.tight_layout()
# axes = axes.flatten()
#
# for i in range(64):
#     image = X_train[i,:].reshape(28,28)
#     title = str(y_train[i])
#     plot_image(image, title, axes[i])
#
# plt.show()


# steps = np.arange(epochs)
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('loss')
# #ax1.set_title('test loss: %.3f, test accuracy: %.3f' % (test_loss, test_acc))
# ax1.plot(steps, train_losses, label="train loss", color='red')
# #ax1.plot(steps, val_losses, label="val loss", color='green')
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel('accuracy')  # we already handled the x-label with ax1
# #ax2.plot(steps, val_accuracy, label="val acc", color='blue')
# ax2.plot(steps,train_accuracy, label="train acc", color='purple')
#
# fig.legend()
# fig.tight_layout()
# # plt.show()


Z1, A1, Z2, A2 = forward_pass(X_test, w1, b1, w2, b2)

test_acc, test_loss = test(X_test, y_test, w1, b1, w2, b2)
pred_test = np.argmax(A2, axis=1)
np.savetxt("NN_pred.csv", pred_test, delimiter=",", fmt='%d')


# # Visual Check of Predictions
# # ----------------------------
# # create plot template
# fig , axes = plt.subplots(8,8, figsize=(24,10))
# fig.tight_layout()
# axes = axes.flatten()
#
# for i in range(64):
#     image = X_test[i,:].reshape(28,28)  # image
#     title = str(pred_test[i])  # class label
#     plot_image(image, title, axes[i])

# plt.show()


def adversarial_samples(x, y, w1, b1, w2, b2, epsilon=100):
    Z1, A1, Z2, A2 = forward_pass(x, w1, b1, w2, b2)

    dlt = A2 - y
    dA1 = np.matmul(dlt, w2.T)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dw1 = np.matmul(dZ1, w1.T)
    direction = np.sign(dw1)
    return x + epsilon * direction, y


x, y = adversarial_samples(X_test, y_test, w1, b1, w2, b2)

_, _, _, y_pred = forward_pass(x, w1, b1, w2, b2)

pred = np.argmax(y_pred, axis=1)

# # create plot template
# fig , axes = plt.subplots(8,8, figsize=(24,10))
# fig.tight_layout()
# axes = axes.flatten()
#
# for i in range(64):
#     image = x[i,:].reshape(28,28)  # image
#     title = str(pred[i])  # class label
#     plot_image(image, title, axes[i])

# plt.show()

DAT = np.column_stack((np.argmax(y, axis=1), pred_test, pred))  # real label, pred without adversarial, with adversarial
np.savetxt("adversarial.csv", DAT, delimiter=",", fmt='%d')
index = 42
fig, axes = plt.subplots(3)
fig.tight_layout()

for i in [index]:
    plot_image(x[i, :].reshape(28, 28), str(pred_test[i]), axes[0])
    plot_image(X_test[i, :].reshape(28, 28), str(pred[i]), axes[1])
    res = X_test[i, :] - x[i, :]
    plot_image(res.reshape(28, 28), "subtraction", axes[2])

    plt.show()



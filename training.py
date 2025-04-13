import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import product

# 数据加载函数
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 数据预处理函数
def load_data():
    train_data, train_labels = [], []
    for i in range(1, 5):  # 加载 data_batch_1 ~ data_batch_4
        batch = unpickle(f"data_batch_{i}")
        train_data.append(batch[b'data'])
        train_labels.append(batch[b'labels'])
    train_data = np.vstack(train_data)     #train_data 为 40000 * 3072 的矩阵。
    train_labels = np.hstack(train_labels) #train_label 为长度  40000 的向量

    val_batch = unpickle("data_batch_5")  # 加载验证集
    val_data = val_batch[b'data']
    val_labels = np.array(val_batch[b'labels'])

    test_batch = unpickle("test_batch")  # 加载测试集
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

# 激活函数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 损失函数
def cross_entropy_plus_L2_regularization_loss(predictions, labels, weights, reg_strength):
    m = labels.shape[0]
    negative_log_likelihood = -np.log(predictions[range(m), labels])
    loss = np.sum(negative_log_likelihood) / m
    reg_loss = reg_strength * np.sum(weights ** 2)
    return loss + reg_loss

# 计算准确率
def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

# 初始化权重
def initialize_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(980005) #My Student ID number
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

# 前向传播
def forward_pass(X, W1, b1, W2, b2, activation_func):
    Z1 = X @ W1 + b1
    A1 = activation_func(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 反向传播
def backward_pass(X, Y, Z1, A1, A2, W1, W2, reg_strength, activation_func):
    m = X.shape[0]
    dZ2 = A2
    dZ2[range(m), Y] -= 1
    dZ2 /= m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    if activation_func == relu:
        dZ1 = dA1 * (Z1 > 0)
    elif activation_func == sigmoid:
        dZ1 = dA1 * A1 * (1 - A1)

    dW1 = X.T @ dZ1 + 2 * reg_strength * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# 训练函数
def train_nn(train_data, train_labels, val_data, val_labels, input_dim, hidden_dim, output_dim, activation_func, 
             batch_size, epochs, learning_rate, reg_strength, full_train):
    print(f"Training with hidden_dim: {hidden_dim}, learning_rate: {learning_rate:.3f}, reg_strength: {reg_strength:.6f}")
    learning_rate_lower_bound = 0.1 * learning_rate
    W1, b1, W2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)
    train_loss_history, val_loss_history, val_acc_history = [], [], []
    lr = learning_rate
    for epoch in range(epochs):
        # 打乱数据
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        # 学习率衰减
        lr = max(lr * learning_rate_decay, learning_rate_lower_bound)

        for i in range(0, train_data.shape[0], batch_size):
            X_batch = train_data[i:i + batch_size]
            Y_batch = train_labels[i:i + batch_size]

            # 前向传播
            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2, activation_func)

            # 反向传播
            dW1, db1, dW2, db2 = backward_pass(X_batch, Y_batch, Z1, A1, A2, W1, W2, reg_strength, activation_func)

            # 更新权重
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # # 计算训练集和验证集的损失与准确率
        _, _, _, train_preds = forward_pass(train_data, W1, b1, W2, b2, activation_func)
        train_loss = cross_entropy_plus_L2_regularization_loss(train_preds, train_labels, W1, reg_strength)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        if(not(full_train)):      
            _, _, _, val_preds = forward_pass(val_data, W1, b1, W2, b2, activation_func)
            val_loss = cross_entropy_plus_L2_regularization_loss(val_preds, val_labels, W1, reg_strength)
            val_acc = accuracy(val_preds, val_labels)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    if(not(full_train)):
        return train_loss_history, val_loss_history, val_acc_history
    else:
        return W1, b1, W2, b2

# 可视化函数
def plot_metrics(train_loss, val_loss, val_acc):
    epochs = len(train_loss)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss, label="Train Loss")
    plt.plot(range(epochs), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Curve")

    plt.show()

# 主函数
if __name__ == "__main__":

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()

    # 数据归一化
    train_data = train_data / 255.0
    val_data = val_data / 255.0
    test_data = test_data / 255.0

    # 固定参数设置
    input_dim = 3072
    output_dim = 10
    batch_size = 64
    epochs = 16
    learning_rate_decay = 0.95

    '''训练'''

    # 超参数设置
    hidden_dims = [256, 512, 1024, 2048]  
    learning_rates = np.arange(0.005, 0.055, 0.005) 
    reg_strengths = [0.001, 0.002, 0.004, 0.008, 0.016]  
    activation_funcs = [relu, sigmoid]

    # 记录最佳结果
    best_accuracy = 0.0
    best_params = None

    for hidden_dim, learning_rate, reg_strength, activation_func in product(hidden_dims, learning_rates, reg_strengths, activation_funcs):
        train_loss_history, val_loss_history, val_acc_history = train_nn(train_data, train_labels, val_data, val_labels, input_dim, hidden_dim, 
                                                     output_dim, activation_func, batch_size, epochs, learning_rate, reg_strength, full_train=False)   
        val_acc = val_acc_history[-1]
        print(f"hidden_dim: {hidden_dim}, learning_rate: {learning_rate:.3f}, reg_strength: {reg_strength:.6f}, Validation Accuracy: {val_acc:.4f}")
        if(val_acc > best_accuracy):
            best_accuracy = val_acc
            best_params = {'hidden_dim': hidden_dim,
                'learning_rate': learning_rate,
                'reg_strength':  reg_strength,
            }

    print(f"Best Validation Accuracy: {best_accuracy:.4f} with params: {best_params}")


    # '''测试'''
    # hidden_dim = 1024
    # learning_rate = 0.04
    # reg_strength = 0.001
    # activation_func = relu
    # full_train_data = np.vstack((train_data, val_data))
    # full_train_labels = np.hstack((train_labels, val_labels))
    # W1, b1, W2, b2 = train_nn(full_train_data, full_train_labels, [], [], input_dim, hidden_dim, output_dim, 
    #     activation_func,batch_size, epochs, learning_rate, reg_strength, full_train=True)
    # _, _, _, test_preds = forward_pass(test_data, W1, b1, W2, b2, activation_func)
    # test_acc = accuracy(test_preds, test_labels)
    # print(f"Test Accuracy: {test_acc:.4f}")
    # np.savez('weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)

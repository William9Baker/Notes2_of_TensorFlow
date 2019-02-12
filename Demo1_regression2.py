import tensorflow as tf

def myregression():
    # 1、准备数据，x 特征值 [100, 1]   （100个样本，10个特征）
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    # weight也是一个op
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="weight")
    bias = tf.Variable(0.0, name="b")    # 初始化为0.0

    y_predict = tf.matmul(x, weight) + bias

    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    initializer_op = tf.global_variables_initializer()  # 初始化变量weight和bias

    with tf.Session() as sess:
        sess.run(initializer_op)
        print("随机初始化的参数权重为: %f, 偏置为: %f" % (weight.eval(), bias.eval()))
       # 循环训练
        for i in range(300):
            sess.run(train_op)
            print("第%d次优化的参数权重为: %f, 偏置为: %f" % (i, weight.eval(), bias.eval()))

    return None

if __name__ == "__main__":
    myregression()
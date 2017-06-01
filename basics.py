import tensorflow as tf


def add_constant():
    node1 = tf.constant(3.0)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)  # node1 + node2 is equivalent

    sess = tf.Session()
    print("Add Constant")
    print(sess.run(node3))


def add_with_placeholder():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add_node = a + b

    sess = tf.Session()
    print("Add with placeholder")
    print(sess.run(add_node, {a: [1.0, 2.0, 3.0], b: [0.5, 0.6, 0.7]}))


def linear_model():
    # A simple linear model
    W = tf.Variable([0.05], tf.float32)
    b = tf.Variable([0.1], tf.float32)
    x = tf.placeholder(tf.float32)
    model = W * x + b

    # Desired output
    y = tf.placeholder(tf.float32)

    # Loss function
    loss = tf.reduce_sum(tf.square(model - y))

    # Training procedure
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Session and initialization
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Learning
    it = 0
    while it < 1000:
        curr_loss = sess.run(loss, {x: [1, 2, 3], y: [0, -1, -2]})
        print("iteration " + str(it) + "\n\tloss = " + str(curr_loss) + "\n")

        if curr_loss < 1e-5:
            break

        sess.run(train, {x: [1, 2, 3], y: [0, -1, -2]})
        it = it + 1

    print("Linear model learned")
    print("W = " + str(sess.run(W)))
    print("b = " + str(sess.run(b)))


def main():
    add_constant()
    add_with_placeholder()
    linear_model()

if __name__ == "__main__":
    main()

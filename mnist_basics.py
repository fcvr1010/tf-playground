import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # MNIST example dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Softmax linear regression model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # True label
    y_star = tf.placeholder(tf.float32, [None, 10])

    # Cross-entropy loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_star, logits=y)
    )

    # Optimizer
    optim = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

    # Init session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Do training
    it = 0
    curr_loss = float("inf")
    while (it < 1000) and (curr_loss > 1e-3):
        # Get a random batch of size 100 from the training data
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed_dict = {x: batch_xs, y_star: batch_ys}

        # Current loss
        curr_loss = sess.run(loss, feed_dict=feed_dict)
        print("Iteration " + str(it) + "\tloss = " + str(curr_loss))

        # One training epoch
        sess.run(optim, feed_dict=feed_dict)
        it = it + 1

    print("\nTraining complete. Residual loss = " + str(curr_loss))

    # Evaluate accuracy on test data
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_star, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy on test data = " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_star: mnist.test.labels})))


if __name__ == "__main__":
    main()

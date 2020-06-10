import os, sys
import tensorflow
import func_CNN

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

my_faces_path = 'E:/FRTM/Face/Face of me/'
other_faces_path = 'E:/FRTM/Face/Face of others/'
size = 64

imgs = []
labs = []

#whatthe()
#cnnTrain()


def cnnTrain():
    out = func_CNN.cnnLayer(x, keep_prob_5, keep_prob_75)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./CNN/CNN_tmp', graph=tf.get_default_graph())

        for n in range(10):
             # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x, y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss,'loss')
                #print('batch_x'+"\n"+str(batch_x)+"\n")
                #print('batch_y'+"\n"+str(batch_y)+"\n")

                #if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                print(n*num_batch+i, acc,'acc')
                # 准确率大于0.98时保存并退出
                if acc > 0.97 and n > 2:
                    #saver.save(sess, './CNN/CNN_model/train_faces.model', global_step=n*num_batch+i)
                    print('save done')
                    sys.exit(0)
        print(f'accuracy is {acc}')

num_batch, keep_prob_5, keep_prob_75, y_, train_x,train_y, batch_size, x, test_x, test_y = func_CNN.whatthe(size, imgs, labs, my_faces_path, other_faces_path)

cnnTrain()
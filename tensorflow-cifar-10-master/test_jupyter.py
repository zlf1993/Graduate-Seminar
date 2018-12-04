import numpy as np
import tensorflow as tf
from time import time
import math


from include.data import get_data_set
from include.model import model, lr



x, y, keep_prob, output, y_pred_cls, global_step, learning_rate = model()
test_x, test_y = get_data_set("test")
global_accuracy = 0
epoch_start = 0
_BATCH_SIZE = 128
_EPOCH = 20
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


saver = tf.train.Saver()
sess = tf.Session()


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except Exception:
    print("\nFailed to restore checkpoint. Initializing variables instead.")

predicted_out = np.zeros(shape=test_y.shape, dtype=np.float32, name='predicted_out')
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
i = 0
while i < len(test_x):
    j = min(i + _BATCH_SIZE, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j], predicted_out[i:j] = sess.run(
        [y_pred_cls, output],
        feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(1), keep_prob: 1.0}
    )
    i = j

vector_embeddings = tf.get_variable('predicted_out', test_y.shape)
with open("metadata.tsv", 'w') as file_metadata:
    for label in enumerate(predicted_class):
        file_metadata.write(label+'\n')
"""
Function: This script is an example of showing how to use progress bar in TnesorFlow
Author: Du Fei
Create Time: 2020/7/18 16:06
"""

import time
import tensorflow as tf

n_epoch = 100
steps_per_epoch = 50

for epoch in range(n_epoch):
    print(f"Epoch:{epoch}/{n_epoch}")

    # Here steps_per_epoch is the total number of steps for current progress bar
    progress_bar = tf.keras.utils.Progbar(steps_per_epoch)
    for step in range(steps_per_epoch):
        time.sleep(0.05)
        # step is the current step index
        progress_bar.update(step, values=[("loss_test", 0.01)])

    print("")

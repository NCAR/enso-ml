import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import tensorflow as tf
from tensorflow.keras import Model, layers, models
from tensorflow import keras
from contextlib import redirect_stdout
import os
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "gpu_number"


begin_time = datetime.now()


def memory_footprint():
    """Returns memory (in MB) being used by Python process"""
    mem = psutil.Process(os.getpid()).memory_info().rss
    return mem / 1024**2


op_train = "op_training"
op_test = "op_testing"
op_gpu = "option_gpu"
ens_number = "ens"

o_path = "h_dir/output/exp_name/comb_name/ENensemble/"

if op_gpu == "on":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

# =======================================================================
# load training set
# =======================================================================
if op_train == "on":

    # input
    f1 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_tos_all.input_17_models.nc"
    )
    f2 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_thetao_all.input_17_models.nc"
    )
    f3 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_uas_all.input_17_models.nc"
    )
    f4 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_vas_all.input_17_models.nc"
    )

    sst = f1.variables["sst"][:, :, :, :]
    t300 = f2.variables["sst"][:, :, :, :]
    uas = f3.variables["sst"][:, :, :, :]
    vas = f4.variables["sst"][:, :, :, :]
    f1.close()
    f2.close()
    f4.close()
    f3.close()
    tdim, zdim, ydim, xdim = sst.shape

    # merge
    tr_x = sst
    # tr_x = np.append(np.append(sst,t300,axis=1),uas,axis=1)
    del sst, t300, uas, vas
    tdim, zdim, ydim, xdim = tr_x.shape
    print("dims t z y x before swapping")
    print(tdim)
    print(zdim)
    print(ydim)
    print(xdim)
    print(tr_x.shape)
    # [tdim,zdim,ydim,xdim] -> [tdim,xdim,ydim,zdim]
    tr_x = np.swapaxes(tr_x, 1, 3)
    print("tr_x after swapping")
    print(tr_x.shape)

    f = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_enso34_all.input_17_models.nc",
        "r",
    )
    tr_y = f.variables["sst"][:, :]
    print(tr_y.shape)
    f.close()

    # =======================================================================
    # load validation set
    # =======================================================================
    # input
    f1 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_tosall_input_validation_winds.nc"
    )
    f2 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_thetaoall_input_validation_winds.nc"
    )
    f3 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_uasall_input_validation_winds.nc"
    )
    f4 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_vasall_input_validation_winds.nc"
    )

    sst = f1.variables["sst"][:, :, :, :]
    t300 = f2.variables["sst"][:, :, :, :]
    uas = f3.variables["sst"][:, :, :, :]
    vas = f4.variables["sst"][:, :, :, :]
    # f.close()
    val_tdim, zdim, ydim, xdim = sst.shape
    val_x = sst
    # val_x = np.append(np.append(sst,t300,axis=1),uas,axis=1)
    del sst, t300, uas, vas

    val_tdim, zdim, ydim, xdim = val_x.shape

    # [val_tdim,zdim,ydim,xdim] -> [val_tdim,xdim,ydim,zdim]
    val_x = np.swapaxes(val_x, 1, 3)

    # label [val_tdim,23]
    f = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/1861_2001_enso34_all_input_validation_winds.nc",
        "r",
    )
    val_y = f.variables["sst"][:, :]
    f.close()

# =======================================================================
# load test set
# =======================================================================
if op_test == "on":

    # input
    f = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/godas.input.1980_2014.nc"
    )
    f2 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/uas_anom_tau320CR_1980_2014.nc"
    )
    f3 = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/vas_anom_tau320CR_1980_2014.nc"
    )
    sst = f.variables["sst"][:, :, :, :]
    t300 = f.variables["hc300"][:, :, :, :]
    uas = f.variables["sst"][:, :, :, :]
    vas = f.variables["sst"][:, :, :, :]
    f.close()
    test_tdim, _, ydim, xdim = sst.shape
    test_x = sst
    # test_x  = np.append(np.append(sst,t300,axis=1),uas,axis=1)
    del sst, t300
    test_tdim, zdim, ydim, xdim = test_x.shape
    # [test_tdim,zdim,ydim,xdim] -> [test_tdim,xdim,ydim,zdim]
    # [test_tdim,zdim,ydim,xdim] -> [test_tdim,xdim,ydim,zdim]
    test_x = np.swapaxes(test_x, 1, 3)
    # label
    f = Dataset(
        "/gws/nopw/j04/aopp/colfescu/test_ml_ham_et_al/data/cmip5/final_data/godas.label.1980_2014.nc"
    )
    print(f.variables)
    test_y = f.variables["sst"][:, :, 0, 0]
    f.close()
# =======================================================================


def conv_set(X, n_feat, k_size, act, stride=1):

    conv = layers.Conv2D(
        n_feat, k_size, activation=act, padding="same", strides=stride
    )(X)

    return conv


def dense_set(X, n_feat, act):

    dense = layers.Dense(n_feat, activation=act)(X)

    return dense


def max_pool(X):

    pool = layers.MaxPool2D((2, 2), strides=2, padding="same")(X)

    return pool


# =======================================================================

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=200,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=o_path + "model.hdf5",
        monitor="val_loss",
        save_best_only=True,
    ),
]

inputs = tf.keras.Input(shape=(xdim, ydim, zdim))

# conv 1
conv1 = conv_set(inputs, conv_f, [8, 4], "tanh")

# pool1
pool1 = max_pool(conv1)

# conv2
conv2 = conv_set(pool1, conv_f, [4, 2], "tanh")

# pool2
pool2 = max_pool(conv2)

# conv3
conv3 = conv_set(pool2, conv_f, [4, 2], "tanh")

# flatten
flat = layers.Flatten()(conv3)

# dense 1
dense1 = dense_set(flat, dens_f, "tanh")

# dense 2
dense2 = dense_set(dense1, dens_f, "tanh")

# output1
output1 = dense_set(dense2, 23, None)

# output2
output2 = dense_set(dense2, 12, "softmax")


# model
model = Model(inputs=inputs, outputs=[output1, output2])

if op_train == "on":

    # compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss=["mse", "categorical_crossentropy"],
        loss_weights=[0.8, 0.2],
    )

    # verbose = 0 and epochs is 1000 in the initial script
    # run
    history = model.fit(
        tr_x,
        [tr_y, tr_z],
        batch_size=512,
        epochs=100,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(val_x, [val_y, val_z]),
    )
    # history = model.fit(tr_x, [tr_y, tr_z],validation_split=0.20, epochs=150, batch_size=10, verbose=1)

    model.save(o_path + "model_last.hdf5")

    history_dict = history.history
    tr_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    # save the model summary, logs
    with open(o_path + "model_summary.md", "w") as f:
        with redirect_stdout(f):
            model.summary()

    tr_loss = np.array(tr_loss)
    val_loss = np.array(val_loss)

    tr_loss.astype("float32").tofile(o_path + "tr_loss.gdat")
    val_loss.astype("float32").tofile(o_path + "val_loss.gdat")

if op_test == "on":

    # load best model
    model = models.load_model(o_path + "model.hdf5")

    # prediction
    y_hat, z_hat = model.predict(test_x)
    y_hat, z_hat = np.array(y_hat), np.array(z_hat)

    # save Nino3.4
    y_hat.astype("float32").tofile(o_path + "nino34.gdat")

    ctl = open(o_path + "nino34.ctl", "w")
    ctl.write("dset ^nino34.gdat\n")
    ctl.write("undef -9.99e+08\n")
    ctl.write("xdef   1  linear   0.  2.5\n")
    ctl.write("ydef   1  linear -90.  2.5\n")
    ctl.write("zdef  23  linear 1 1\n")
    ctl.write("tdef " + str(test_tdim) + "  linear jan1980 1yr\n")
    ctl.write("vars   1\n")
    ctl.write("p   23   1  pr\n")
    ctl.write("ENDVARS\n")
    ctl.close()

    # save calendar month
    z_hat.astype("float32").tofile(o_path + "month.gdat")

    ctl = open(o_path + "month.ctl", "w")
    ctl.write("dset ^month.gdat\n")
    ctl.write("undef -9.99e+08\n")
    ctl.write("xdef   1  linear   0.  2.5\n")
    ctl.write("ydef   1  linear -90.  2.5\n")
    ctl.write("zdef  12  linear 1 1\n")
    ctl.write("tdef " + str(test_tdim) + "  linear jan1980 1yr\n")
    ctl.write("vars   1\n")
    ctl.write("p   12   1  pr\n")
    ctl.write("ENDVARS\n")
    ctl.close()

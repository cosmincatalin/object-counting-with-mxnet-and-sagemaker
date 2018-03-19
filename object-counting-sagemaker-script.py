import base64
import json
import logging
from pickle import load

import mxnet as mx
import numpy as np
import pip
from mxnet import autograd, nd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss
from mxnet.gluon.nn import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Sequential
from mxnet.initializer import Xavier


def install(package):
    pip.main(['install', package])


install("opencv-python")
import cv2

logging.basicConfig(level=logging.INFO)


# https://docs.aws.amazon.com/sagemaker/latest/dg/mxnet-training-inference-code-template.html

def train(hyperparameters, channel_input_dirs, num_gpus, hosts):
    batch_size = hyperparameters.get("batch_size", 64)
    epochs = hyperparameters.get("epochs", 3)

    mx.random.seed(42)

    training_dir = channel_input_dirs['training']

    logging.info("Loading data from {}".format(training_dir))

    with open("{}/train/data.p".format(training_dir), "rb") as pickle:
        train_nd = load(pickle)
    with open("{}/validation/data.p".format(training_dir), "rb") as pickle:
        validation_nd = load(pickle)

    train_data = gluon.data.DataLoader(train_nd, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(validation_nd, batch_size, shuffle=True)

    net = Sequential()
    # http: // gluon.mxnet.io / chapter03_deep - neural - networks / plumbing.html  # What's-the-deal-with-name_scope()?
    with net.name_scope():
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(MaxPool2D(pool_size=(2, 2)))
        net.add(Dropout(.25))
        net.add(Flatten())
        net.add(Dense(1))

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    # Also known as Glorot
    net.collect_params().initialize(Xavier(magnitude=2.24), ctx=ctx)
    # Calculates the mean squared error between pred and label.
    loss = L2Loss()

    # kvstore type for multi - gpu and distributed training.
    if len(hosts) == 1:
        kvstore = "device" if num_gpus > 0 else "local"
    else:
        kvstore = "dist_device_sync'" if num_gpus > 0 else "dist_sync"

    trainer = Trainer(net.collect_params(), optimizer="adam", kvstore=kvstore)

    smoothing_constant = .01

    for e in range(epochs):
        moving_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss_result = loss(output, label)
            loss_result.backward()
            trainer.step(batch_size)

            curr_loss = nd.mean(loss_result).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
        trn_total, trn_detected = measure_performance(net, ctx, train_data)
        validation_total, validation_detected = measure_performance(net, ctx, validation_data)
        logging.info("Epoch {}: loss: {:0.4f} Test accuracy: {:0.2f} Validation accuracy: {:0.2f}"
                     .format(e, moving_loss, trn_detected / trn_total, validation_detected / validation_total))

    return net


def measure_performance(model, ctx, data_iter):
    raw_predictions = np.array([])
    rounded_predictions = np.array([])
    actual_labels = np.array([])
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(data)
        predictions = nd.round(output)
        raw_predictions = np.append(raw_predictions, output.asnumpy().squeeze())
        rounded_predictions = np.append(rounded_predictions, predictions.asnumpy().squeeze())
        actual_labels = np.append(actual_labels, label.asnumpy().squeeze())

    results = np.concatenate((raw_predictions.reshape((-1, 1)),
                              rounded_predictions.reshape((-1, 1)),
                              actual_labels.reshape((-1, 1))), axis=1)
    detected = 0
    i = -1
    for i in range(int(results.size / 3)):
        if results[i][1] == results[i][2]:
            detected += 1
    return i + 1, detected


def save(net, model_dir):
    y = net(mx.sym.var("data"))
    y.save("{}/model.json".format(model_dir))
    net.collect_params().save("{}/model.params".format(model_dir))


def model_fn(model_dir):
    with open("{}/model.json".format(model_dir), "r") as model_file:
        model_json = model_file.read()
    outputs = mx.sym.load_json(model_json)
    inputs = mx.sym.var("data")
    param_dict = gluon.ParameterDict("model_")
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    # We will serve the model on CPU
    net.load_params("{}/model.params".format(model_dir), ctx=mx.cpu())
    return net


# noinspection PyUnusedLocal
def transform_fn(model, input_data, content_type, accept):
    if content_type == "application/png":
        img = img2arr(input_data)
        response = model(img).asnumpy().ravel().tolist()
        return json.dumps(response), accept
    elif content_type == "application/json":
        json_array = json.loads(input_data, encoding="utf-8")
        imgs = [img2arr(base64img) for base64img in json_array]
        imgs = np.concatenate(imgs)
        imgs = nd.array(imgs)
        response = model(imgs)
        response = nd.round(response)
        response = response.asnumpy()
        response = response.ravel()
        response = response.tolist()
        return json.dumps(response), accept
    else:
        raise ValueError("Cannot decode input to the prediction.")


def img2arr(base64img):
    img = base64.b64decode(base64img)
    img = np.asarray(bytearray(img), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img = mx.nd.array(img)
    img = mx.nd.transpose(img, (2, 0, 1))
    img = img / 255
    img = img.reshape((1, 3, 128, 128))
    img = img.asnumpy()
    return img

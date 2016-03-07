require(mxnet)

## running device
dev <- mx.gpu(0)

batch.size <- 64
data.shape <- c( 36, 36, 3)
## training data info for learning rate reduction
num.examples <- 22750
epoch_size <- num.examples / batch.size
lr_factor_epoch <- 15
## model saving parameter
model_prefix <- "./models/sample_net"
data_dir<-"data48/"
## train data iterator
train <- mx.io.ImageRecordIter(
        path_imgrec = paste(data_dir,"tr.rec",sep=""),
        mean_r      = 128,
        mean_g      = 128,
        mean_b      = 128,
        scale       = 0.0078125,
        max_aspect_ratio = 0.35,
        data.shape  = data.shape,
        batch.size  = batch.size,
        rand_crop   = TRUE,
        rand_mirror = TRUE)

## validate data iterator
val <- mx.io.ImageRecordIter(
        path_imgrec = paste(data_dir,"va.rec",sep=""),
        mean_r      = 128,
        mean_b      = 128,
        mean_g      = 128,
        scale       = 0.0078125,
        rand_crop   = FALSE,
        rand_mirror = FALSE,
        data.shape  = data.shape,
        batch.size  = batch.size)

## load network definition
## network definition
## stage 1
data <- mx.symbol.Variable("data")
conv1 <- mx.symbol.Convolution(data, kernel=c(5, 5), num.filter=32, pad=c(2, 2))
act1 <- mx.symbol.Activation(conv1, act.type="relu")
conv2 <- mx.symbol.Convolution(act1, kernel=c(5, 5), num.filter=64, pad=c(2, 2))
act2 <- mx.symbol.Activation(conv2, act.type="relu")
pool1 <- mx.symbol.Pooling(act2, pool_type="max", kernel=c(3, 3), stride=c(2, 2))
## stage 2
conv3 <- mx.symbol.Convolution(pool1, kernel=c(3, 3), num.filter=64, pad=c(1, 1))
act3 <- mx.symbol.Activation(conv3, act.type="relu")
conv4 <- mx.symbol.Convolution(act3, kernel=c(3, 3), num.filter=64, pad=c(1, 1))
act4 <- mx.symbol.Activation(conv4, act.type="relu")
conv5 <- mx.symbol.Convolution(act4, kernel=c(3, 3), num.filter=128, pad=c(1, 1))
act5 <- mx.symbol.Activation(conv5, act.type="relu")
pool2 <- mx.symbol.Pooling(act5, pool_type="max", kernel=c(3, 3), stride=c(2, 2))
## stage 3
conv6 <- mx.symbol.Convolution(pool2, kernel=c(3, 3), num.filter=256, pad=c(1, 1))
act6 <- mx.symbol.Activation(conv6, act.type="relu")
conv7 <- mx.symbol.Convolution(act6, kernel=c(3, 3), num.filter=256, pad=c(1, 1))
act7 <- mx.symbol.Activation(data=conv7, act.type="relu")
pool3 <- mx.symbol.Pooling(act7, pool_type="avg", kernel=c(9, 9), stride=c(1, 1))
## stage 4
flat1 <- mx.symbol.Flatten(pool3)
do1 <- mx.symbol.Dropout(flat1, p=0.25)
fc1 <- mx.symbol.FullyConnected(do1, num.hidden=121)
softmax <- mx.symbol.SoftmaxOutput(fc1, name='softmax')

mx.set.seed(0)

logger <- mx.metric.logger$new()
## Model parameter
## This model will reduce learning rate by factor 0.1 for every 15 epoch

model <- mx.model.FeedForward.create(
    symbol             = softmax,
    X                  = train,
    eval.data          = val,
    epoch.end.callback = mx.callback.save.checkpoint(model_prefix),
##    epoch.end.callback = mx.callback.log.train.metric(1,logger),
    batch.end.callback = mx.callback.log.train.metric(50),
    ctx                = dev,
    num.round          = 4,
    learning.rate      = 0.01,
    momentum           = 0.9,
    wd                 = 0.0001,
    initializer = mx.init.Xavier(
                                        rnd_type = "gaussian",
                                        factor_type = "in",
                                        magnitude = 2.34),
   clip_gradient = 5,
   lr_scheduler=FactorScheduler(step=epoch_size * lr_factor_epoch, factor = 0.1),
  eval.metric=mx.metric.accuracy
)

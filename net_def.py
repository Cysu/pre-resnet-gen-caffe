import os
import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser

CAFFE_ROOT = osp.join(osp.dirname(__file__), '..', 'caffe')
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
from caffe.proto import caffe_pb2


def _get_include(phase):
    inc = caffe_pb2.NetStateRule()
    if phase == 'train':
        inc.phase = caffe_pb2.TRAIN
    elif phase == 'test':
        inc.phase = caffe_pb2.TEST
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return inc


def _get_param(num_param):
    if num_param == 1:
        # only weight
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1
        param.decay_mult = 1
        return [param]
    elif num_param == 2:
        # weight and bias
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1
        param_w.decay_mult = 1
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 2
        param_b.decay_mult = 0
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))


def _get_transform_param(phase):
    param = caffe_pb2.TransformationParameter()
    param.crop_size = 224
    param.mean_value.extend([104, 117, 123])
    param.force_color = True
    if phase == 'train':
        param.mirror = True
    elif phase == 'test':
        param.mirror = False
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return param


def Data(name, tops, source, batch_size, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Data'
    layer.top.extend(tops)
    layer.data_param.source = source
    layer.data_param.batch_size = batch_size
    layer.data_param.backend = caffe_pb2.DataParameter.LMDB
    layer.include.extend([_get_include(phase)])
    layer.transform_param.CopyFrom(_get_transform_param(phase))
    return layer


def Conv(name, bottom, num_output, kernel_size, stride, pad):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = 'msra'
    layer.convolution_param.bias_term = False
    layer.param.extend(_get_param(1))
    return layer


def Act(name, bottom):
    top_name = name
    # BN
    bn_layer = caffe_pb2.LayerParameter()
    bn_layer.name = name + '_bn'
    bn_layer.type = 'BatchNorm'
    bn_layer.bottom.extend([bottom])
    bn_layer.top.extend([top_name])
    # Scale
    scale_layer = caffe_pb2.LayerParameter()
    scale_layer.name = name + '_scale'
    scale_layer.type = 'Scale'
    scale_layer.bottom.extend([top_name])
    scale_layer.top.extend([top_name])
    scale_layer.scale_param.filler.value = 1
    scale_layer.scale_param.bias_term = True
    scale_layer.scale_param.bias_filler.value = 1
    # ReLU
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_relu'
    relu_layer.type = 'ReLU'
    relu_layer.bottom.extend([top_name])
    relu_layer.top.extend([top_name])
    return [bn_layer, scale_layer, relu_layer]


def Pool(name, bottom, pooling_method, kernel_size, stride, pad):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    if pooling_method == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_method == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    else:
        raise ValueError("Unknown pooling method {}".format(pooling_method))
    layer.pooling_param.kernel_size = kernel_size
    layer.pooling_param.stride = stride
    layer.pooling_param.pad = pad
    return layer


def Linear(name, bottom, num_output):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_filler.value = 0
    layer.param.extend(_get_param(2))
    return layer


def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer


def ResBlock(name, bottom, dim, stride, block_type=None):
    layers = []
    if block_type == 'no_preact':
        res_bottom = bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_proj', res_bottom, dim*4, 1, stride, 0))
        shortcut_top = layers[-1].top[0]
    elif block_type == 'both_preact':
        layers.extend(Act(name + '_pre', bottom))
        res_bottom = layers[-1].top[0]
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_proj', res_bottom, dim*4, 1, stride, 0))
        shortcut_top = layers[-1].top[0]
    else:
        shortcut_top = bottom
        # preact at residual branch
        layers.extend(Act(name + '_pre', bottom))
        res_bottom = layers[-1].top[0]
    # residual branch: conv1 -> conv1_act -> conv2 -> conv2_act -> conv3
    layers.append(Conv(name + '_conv1', res_bottom, dim, 1, 1, 0))
    layers.extend(Act(name + '_conv1', layers[-1].top[0]))
    layers.append(Conv(name + '_conv2', layers[-1].top[0], dim, 3, stride, 1))
    layers.extend(Act(name + '_conv2', layers[-1].top[0]))
    layers.append(Conv(name + '_conv3', layers[-1].top[0], dim*4, 1, 1, 0))
    # elementwise addition
    layers.append(Add(name, [shortcut_top, layers[-1].top[0]]))
    return layers


def ResLayer(name, bottom, num_blocks, dim, stride, layer_type=None):
    assert num_blocks >= 1
    _get_name = lambda i: '{}_res{}'.format(name, i)
    layers = []
    first_block_type = 'no_preact' if layer_type == 'first' else 'both_preact'
    layers.extend(ResBlock(_get_name(1), bottom, dim, stride, first_block_type))
    for i in xrange(2, num_blocks+1):
        layers.extend(ResBlock(_get_name(i), layers[-1].top[0], dim, 1))
    return layers


def Loss(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'SoftmaxWithLoss'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer


def Accuracy(name, bottoms, top_k):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Accuracy'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.accuracy_param.top_k = top_k
    layer.include.extend([_get_include('test')])
    return layer


def create_model(depth):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    configs = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }
    num = configs[depth]
    layers = []
    layers.append(Data('data', ['data', 'label'],
                       'examples/imagenet/ilsvrc12_train_lmdb', 32, 'train'))
    layers.append(Data('data', ['data', 'label'],
                       'examples/imagenet/ilsvrc12_val_lmdb', 25, 'test'))
    layers.append(Conv('conv1', 'data', 64, 7, 2, 3))
    layers.extend(Act('conv1', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('conv2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('conv3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('conv4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('conv5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Act('conv5', layers[-1].top[0]))
    layers.append(Pool('pool5', layers[-1].top[0], 'ave', 7, 1, 0))
    layers.append(Linear('fc', layers[-1].top[0], 1000))
    layers.append(Loss('loss', ['fc', 'label']))
    layers.append(Accuracy('accuracy_top1', ['fc', 'label'], 1))
    layers.append(Accuracy('accuracy_top5', ['fc', 'label'], 5))
    model.layer.extend(layers)
    return model


def main(args):
    model = create_model(args.depth)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__),
            'resnet{}_trainval.prototxt'.format(args.depth))
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--depth', type=int, default=200,
                        choices=[50, 101, 152, 200])
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    main(args)

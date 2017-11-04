from chainer import ChainList, optimizers, serializers
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class _CRF(ChainList):
    def __init__(self):
        super(_CRF, self).__init__(L.ConvolutionND(1, 2, 2, 1, nobias = True))

    def __call__(self, x, y):
        z = F.softmax(-y)

        for i in range(10):
            z = -y - self[0](F.batch_matmul(z, x))

            if i < 4:
                z = F.softmax(z)

        return z

class _ResidualBlock(ChainList):
    def __init__(self, dilate):
        super(_ResidualBlock, self).__init__(L.DilatedConvolution2D(61, 122, (1, 2), dilate = dilate),
                                             L.Convolution2D(61, 573, 1))

    def __call__(self, x):
        y = F.split_axis(self[0](F.pad(x, ((0, 0), (0, 0), (0, 0), (self[0].dilate[1], 0)), 'constant')), 2, 1)
        y = F.split_axis(self[1](F.sigmoid(y[0]) * F.tanh(y[1])), (61,), 1)

        return x + y[0], y[1]

class _WaveNet(ChainList):
    def __init__(self):
        links = (L.Convolution2D(61, 61, (1, 2)),)
        links += tuple(_ResidualBlock((1, 2 ** (i % 6))) for i in range(6))
        links += (L.Convolution2D(512, 512, 1), L.Convolution2D(512, 3843, 1))

        super(_WaveNet, self).__init__(*links)

    def __call__(self, x):
        y = (self[0](F.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)), 'constant')),)
        z = 0

        for i in range(1, len(self) - 2):
            y = self[i](y[0])
            z += y[1]

        y, z = F.split_axis(self[-1](F.relu(self[-2](z))), (3721,), 1)

        return F.reshape(y, (y.shape[0], 61, 61, y.shape[3])), \
               F.reshape(z, (z.shape[0], 2, 61, z.shape[3]))

class WaveCRF(object):
    def __init__(self):
        self.log = {('test', 'accuracy'): (), ('test', 'loss'): (), ('training', 'accuracy'): (),
                    ('training', 'loss'): ()}
        self.model = ChainList(_WaveNet(), _CRF())
        self.optimizer = optimizers.Adam(0.0002, 0.5)

        self.optimizer.setup(self.model)

    def __call__(self, x):
        k, psi_u = self.model[0](x)
        Q_hat = self.model[1](F.reshape(F.transpose(k, (0, 3, 1, 2)), (-1, 61, 61)),
                              F.reshape(F.transpose(psi_u, (0, 3, 1, 2)), (-1, 2, 61)))

        return F.transpose(F.reshape(Q_hat, (x.shape[0], x.shape[3], 2, 61)), (0, 2, 3, 1))

    @classmethod
    def load(cls, directory):
        self = cls()
        self.log = np.load('{}/log.npy'.format(directory))

        serializers.load_npz('{}/model.npz'.format(directory), self.model)
        serializers.load_npz('{}/optimizer.npz'.format(directory), self.optimizer)

        return self

    def save(self, directory):
        np.save('{}/log.npy'.format(directory), self.log)
        serializers.save_npz('{}/model.npz'.format(directory), self.model)
        serializers.save_npz('{}/optimizer.npz'.format(directory), self.optimizer)

    def test(self, Q, x):
        with chainer.using_config('train', False):
            Q_hat = self(x)
            loss = F.softmax_cross_entropy(Q_hat, Q)

            self.log['test', 'accuracy'] += (float(F.accuracy(Q_hat, Q).data),)
            self.log['test', 'loss'] += (float(loss.data),)

    def train(self, Q, x):
        Q_hat = self(x)
        loss = F.softmax_cross_entropy(Q_hat, Q)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

        self.log['training', 'accuracy'] += (float(F.accuracy(Q_hat, Q).data),)
        self.log['training', 'loss'] += (float(loss.data),)



# Cnn变种算法

## Gannet（Generative Adversarial Networks）

### 1.基本原理

最优化价值函的 G 和 D 以求解极小极大博弈:

$ \min_{G}\max_{D}V(D,G)=E_{x~P_{data}(x)}[logD(x)]+E_{x~P_z(z)}[log(1-D(G(z)))] $



![img](https://img-blog.csdn.net/2018100901061418?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NfY2h1eGlu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

###  2.优点

- GAN 模型只用到了反向传播,而不需要马尔科夫链
- 训练时不需要对隐变量做推断
- 理论上,只要是可微分函数都可以用于构建 D 和 G ,因为能够与深度神经网络结合做深度生成式模型
- G 的参数更新不是直接来自数据样本,而是使用来自 D 的反向传播
- 相比其他生成模型（VAE、玻尔兹曼机），可以生成更好的生成样本
- GAN 是一种半监督学习模型，对训练集不需要太多有标签的数据；
- 没有必要遵循任何种类的因子分解去设计模型,所有的生成器和鉴别器都可以正常工作

### 3. 缺点

- 可解释性差,生成模型的分布 `Pg(G)`没有显式的表达
- 比较难训练, D 与 G 之间需要很好的同步,例如 D 更新 k 次而 G 更新一次
- 训练 GAN 需要达到纳什均衡,有时候可以用梯度下降法做到,有时候做不到.我们还没有找到很好的达到纳什均衡的方法,所以训练 GAN 相比 VAE 或者 PixelRNN 是不稳定的,但我认为在实践中它还是比训练玻尔兹曼机稳定的多.
- 它很难去学习生成离散的数据,就像文本
- 相比玻尔兹曼机,GANs 很难根据一个像素值去猜测另外一个像素值,GANs 天生就是做一件事的,那就是一次产生所有像素,你可以用 BiGAN 来修正这个特性,它能让你像使用玻尔兹曼机一样去使用 Gibbs 采样来猜测缺失值
- 训练不稳定，G 和 D 很难收敛；
- 训练还会遭遇梯度消失、模式崩溃的问题
- 缺乏比较有效的直接可观的评估模型生成效果的方法

```python
#Gan
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import tensorflow as tf
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
from glob import glob
import numpy as np

class GAN():
    def __init__(self):
        #RGB图像作为输入,剪切后大小
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 3
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        #optimizer = Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)
        #optimizer = SGD(lr=0.01,momentum=0.5)
        optimizer = RMSprop(lr=0.0002,rho=0.9)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',optimizer=optimizer)
        
        z = Input(shape=(100,))
        img = self.generator(z)
        
        self.discriminator.trainable = False
        
        valid = self.discriminator(img)
        
        self.combined = Model(z,valid)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)
    
    def build_generator(self):
        noise_shape = (100,)
        model = Sequential()
        model.add(Dense(256,input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=noise_shape)
        img = model(noise)
        return Model(noise,img)
    
    def build_discriminator(self):
        img_shape = (self.img_rows,self.img_cols,self.channels)
        model = Sequential()
        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img,validity)
    def get_image(self,image_path,width,height,mode):
        image = Image.open(image_path)#并非矩阵，通过np.array(image)变为矩阵
        if image.size != (width,height):#剪切后图片大小
            face_width = face_height = 100
            j = (image.size[0] - face_width) // 2
            i = (image.size[1] - face_height) // 2
            image = image.crop([j,i,j + face_width, i + face_height])# (宽起点，高起点，宽终点，高终点)
            image = image.resize([width,height])
        return np.array(image.convert(mode))#RGB
    def get_batch(self,image_files,width,height,mode):
        #image_files所有图片路径组成的列表
        #sample_file单张图片
        data_batch = np.array(
            [self.get_image(sample_file,width,height,mode) for sample_file in image_files])
        return data_batch
    def train(self,epochs,batch_size=128,save_interval=50):
        data_dir = 'C:\\Users\\Administrator\\Desktop\\血管分叉\\'
        #x_train所有图片组成的矩阵,(2800,100,100,3)
        x_train = self.get_batch(glob(os.path.join(data_dir,'*.jpg'))[:2800],self.img_rows,self.img_cols,'RGB')
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        half_batch = int(batch_size / 2)
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []
        for epoch in range(epochs):
            idx = np.random.randint(0,x_train.shape[0],half_batch)
            imgs = x_train[idx]#一个批次图片，half_batch随机图片矩阵,(16,100,100,3)
            noise = np.random.normal(0,1,(half_batch,100))
            gen_imgs = self.generator.predict(noise)
            #按照自定的batch训练
            d_loss_real = self.discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
            d_loss  = 0.5 *np.add(d_loss_real,d_loss_fake)#[a,b] a为loss,b为acc
            noise = np.random.normal(0,1,(batch_size,100))
            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise,valid_y)
            print('epochs: %d [D loss: %f, acc: %.2f%%], [G loss: %f]' % (epoch ,d_loss[0], 100 *d_loss[1],g_loss))
            #存储loss
            d_loss_logs_r.append([epoch,d_loss[0]])
            d_loss_logs_f.append([epoch,d_loss[1]])
            g_loss_logs.append([epoch,g_loss])
            
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        d_loss_logs_r_a = np.array(d_loss_logs_r)
        d_loss_logs_f_a = np.array(d_loss_logs_f)
        g_loss_logs_a = np.array(g_loss_logs)
        #plt.plot(epoch,loss值大小)
        plt.plot(d_loss_logs_r_a[:,0],d_loss_logs_r_a[:,1],label='Discriminator Loss - Real')
        plt.plot(d_loss_logs_f_a[:,0],d_loss_logs_f_a[:,1],label='discriminator Loss - Fake')
        plt.plot(g_loss_logs_a[:,0],g_loss_logs_a[:,1],label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN')
        plt.grid(True)
        plt.show()
    def save_imgs(self,epoch):
        rows,cols = 5, 5
        noise = np.random.normal(0,1,(rows * cols, 100))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = (1/2.5) * gen_imgs + 0.5
        fig,axs = plt.subplots(rows,cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        if not os.path.exists('gen'):
            os.makedirs('gen')
            
        fig.savefig('gen/%d.png' %epoch)
        plt.close()
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=50,batch_size=40,save_interval=10)
```

```python
#DcGan
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import tensorflow as tf
from scipy.misc import imread, imsave

import matplotlib.pyplot as plt

import sys
import os
from PIL import Image
from glob import glob
import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 96
        self.img_cols = 96
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        
        noise_shape = (100,)

        model = Sequential()
        #layer1 (None,100)>>(None,128*24*24)
        model.add(Dense(128 * self.img_rows//4 * self.img_cols//4, activation="relu", input_shape=noise_shape))
        #(None,24*24*128)>>(None,24,24,128)
        model.add(Reshape((self.img_rows//4, self.img_cols//4, 128)))
        model.add(BatchNormalization(momentum=0.8))
        #layer2 (None,24,24,128)>>(None,48,48,128)
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        #layer3 (None,48,48,128)>>(None,96,96,128)
        model.add(UpSampling2D())
        #(None,96,96,128)>>(None,96,96,64)
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        #(None,96,96,64)>>(None,96,96,3)
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        #layer1 (None,96,96,3)>>(None,48,48,32)
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #layer2 (None,48,48,32)>>(None,24,24,64)
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        #(None,24,24,64)>>(None,12,12,128)
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        #(None,12,12,128)>>(None,6,6,256)
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #(None,6,6,128)>>(None,3,3,512)
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #(None,3,3,512)>>(None,4608)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    def get_image(self,image_path,width,height,mode):
        image = Image.open(image_path)
#         print(image.size)
        if image.size != (width,height):
            j = (image.size[0] - self.img_rows) // 2
            i = (image.size[1] - self.img_cols) // 2
            image = image.crop([j,i,j+self.img_rows,i+self.img_cols])
#             image = image.resize(width,height)
            
        return np.array(image.convert(mode))
    def get_batch(self,image_files,width,height,mode):
        data_batch = np.array([self.get_image(sample_file,width,height,mode) for sample_file in image_files])
        return data_batch
        
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        data_dir = 'C:\\Users\\Administrator\\Desktop\\血管分叉\\'
        X_train = self.get_batch(glob(os.path.join(data_dir,'*.jpg'))[:2800],self.img_rows,self.img_cols,'RGB')
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#         X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        
        # Array initialization for logging of the losses
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            # Store the losses
            d_loss_logs_r.append([epoch, d_loss[0]])
            d_loss_logs_f.append([epoch, d_loss[1]])
            g_loss_logs.append([epoch, g_loss])
            
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        d_loss_logs_r_a = np.array(d_loss_logs_r)
        d_loss_logs_f_a = np.array(d_loss_logs_f)
        g_loss_logs_a = np.array(g_loss_logs)

        # At the end of training plot the losses vs epochs
        plt.plot(d_loss_logs_r_a[:,0], d_loss_logs_r_a[:,1], label="Discriminator Loss - Real")
        plt.plot(d_loss_logs_f_a[:,0], d_loss_logs_f_a[:,1], label="Discriminator Loss - Fake")
        plt.plot(g_loss_logs_a[:,0], g_loss_logs_a[:,1], label="Generator Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN')
        plt.grid(True)
        plt.show() 

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
#         gen_imgs = (1/2.5)*gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        if not os.path.exists('血管分叉1'):
            os.makedirs('血管分叉1')
        fig.savefig("血管分叉1/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=500, batch_size=32, save_interval=20)
```

## MobileNets

![MoblieNets](https://img-blog.csdn.net/20170425202723997?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```python
class MobileNet(object):
    def __init__(self, inputs, num_classes=1000, is_training=True,
                 width_multiplier=1, scope="MobileNet"):
        """
        The implement of MobileNet(ref:https://arxiv.org/abs/1704.04861)
        :param inputs: 4-D Tensor of [batch_size, height, width, channels]
        :param num_classes: number of classes
        :param is_training: Boolean, whether or not the model is training
        :param width_multiplier: float, controls the size of model
        :param scope: Optional scope for variables
        """
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multiplier
 
        # construct model
        with tf.variable_scope(scope):
            # conv1
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier),
                         filter_size=3,strides=2)  # ->[N, 112, 112, 32]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=self.is_training))
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                "ds_conv_2") # ->[N, 112, 112, 64]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_3", downsample=True) # ->[N, 56, 56, 128]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_4") # ->[N, 56, 56, 128]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_5", downsample=True) # ->[N, 28, 28, 256]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_6") # ->[N, 28, 28, 256]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_7", downsample=True) # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_8") # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_9")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_10")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_11")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_12")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_13", downsample=True) # ->[N, 7, 7, 1024]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_14") # ->[N, 7, 7, 1024]
            net = avg_pool(net, 7, "avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)
 
    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        """depthwise separable convolution 2D function"""
        num_filters = round(num_filters * width_multiplier)
        strides = 2 if downsample else 1
 
        with tf.variable_scope(scope):
            # depthwise conv2d
            dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            # batchnorm
            bn = bacthnorm(dw_conv, "dw_bn", is_training=self.is_training)
            # relu
            relu = tf.nn.relu(bn)
            # pointwise conv2d (1x1)
            pw_conv = conv2d(relu, "pointwise_conv", num_filters)
            # bn
            bn = bacthnorm(pw_conv, "pw_bn", is_training=self.is_training)
            return tf.nn.relu(bn)
```

##  Faster R-CNN

![img](https://img-blog.csdn.net/20170324121024882)

### 一. 使用VGG16或者其他成熟的图片分类模型提取图片特征(feature map)

### 二. 将图片特征喂入RPN(Region Proposal Network)网络得到proposals (包含第一次回归)

![img](https://img-blog.csdn.net/20170328113414055)

​	  **RPN**网络实际分为2条线，上面一条通过softmax分类anchors获得foreground和background（检测目标是foreground），下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合foreground anchors和bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。



### 三. 将上两步的结果:图片特征和 proposals 喂入RoI Pooling层得到综合的proposals特征

![img](https://img-blog.csdn.net/20180119150020632?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

ROIs Pooling是Pooling层的一种，而且是针对RoIs的Pooling，他的特点是**输入特征图**尺寸不固定，但是**输出特征图**尺寸固定；

#### ROI Pooling的输入

输入有两部分组成： 
1. 特征图：指的是图1中所示的特征图，在Fast RCNN中，它位于RoI Pooling之前，在Faster RCNN中，它是与RPN共享那个特征图，通常我们常常称之为“share_conv”； 
2. rois：在Fast RCNN中，指的是Selective Search的输出；在Faster RCNN中指的是RPN的输出，一堆矩形候选框框，形状为1x5x1x1（4个坐标+索引index），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对原图的（神经网络最开始的输入）

#### ROI Pooling的输出

输出是batch个vector，其中batch的值等于RoI的个数，vector的大小为channel * w * h；RoI Pooling的过程就是将一个个大小不同的box矩形框，都映射成大小固定（w * h）的矩形框；

### 四. 根据poposals特征预测物体的bounding box和物体的类别 (包含第二次回归)

#### bounding box regression原理

如图所示绿色框为飞机的Ground Truth(GT)，红色为提取的foreground anchors，那么即便红色的框被分类器识别为飞机，但是由于红色的框定位不准，这张图相当于没有正确的检测出飞机。所以我们希望采用一种方法对红色的框进行微调，使得foreground anchors和GT更加接近。



![img](https://img-blog.csdn.net/20170321000420426)

缩进对于窗口一般使用四维向量(x, y, w, h)表示，分别表示窗口的中心点坐标和宽高。对于图 10，红色的框A代表原始的Foreground Anchors，绿色的框G代表目标的GT，我们的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'，即：给定A=(Ax, Ay, Aw, Ah)，寻找一种映射f，使得f(Ax, Ay, Aw, Ah)=(G'x, G'y, G'w, G'h)，其中(G'x, G'y, G'w, G'h)≈(Gx, Gy, Gw, Gh)。

![img](https://img-blog.csdn.net/20170321221228658)

*图10*

那么经过何种变换才能从图6中的A变为G'呢？ 比较简单的思路就是:

缩进 1. 先做平移

![img](https://img-blog.csdn.net/20170322104630982)

缩进 2. 再做缩放

![img](https://img-blog.csdn.net/20170322104634390)

缩进观察上面4个公式发现，需要学习的是dx(A)，dy(A)，dw(A)，dh(A)这四个变换。当输入的anchor与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调（注意，只有当anchors和GT比较接近时，才能使用线性回归模型，否则就是复杂的非线性问题了）。对应于Faster RCNN原文，平移量(tx, ty)与尺度因子(tw, th)如下：

![img](https://img-blog.csdn.net/20170614165625267)

缩进接下来的问题就是如何通过线性回归获得dx(A)，dy(A)，dw(A)，dh(A)了。线性回归就是给定输入的特征向量X, 学习一组参数W, 使得经过线性回归后的值跟真实值Y（即GT）非常接近，即Y=WX。对于该问题，输入X是一张经过num_output=1的1x1卷积获得的feature map，定义为Φ；同时还有训练传入的GT，即(tx, ty, tw, th)。输出是dx(A)，dy(A)，dw(A)，dh(A)四个变换。那么目标函数可以表示为：

![img](https://img-blog.csdn.net/20170322110627709)

其中Φ(A)是对应anchor的feature map组成的特征向量，w是需要学习的参数，d(A)是得到的预测值（*表示 x，y，w，h，也就是每一个变换对应一个上述目标函数）。为了让预测值(tx, ty, tw, th)与真实值最小，得到损失函数：

![img](https://img-blog.csdn.net/20170322113250485)

函数优化目标为：

![img](https://img-blog.csdn.net/20170610002805438)

#### Proposal Layer

**Proposal Layer**有3个输入：fg/bg anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的[dx(A)，dy(A)，dw(A)，dh(A)]变换量rpn_bbox_pred，以及im_info；

```python

```

## MASK-RCNN

Mask R-CNN 是一个两阶段的框架，第一个阶段扫描图像并生成提议（proposals，即有可能包含一个目标的区域），第二阶段分类提议并生成边界框和掩码。Mask R-CNN 扩展自 Faster R-CNN，由同一作者提出。Faster R-CNN 是一个流行的目标检测框架，Mask R-CNN 将其扩展为实例分割框架。

有效地检测图像中的目标，同时还能为每个实例生成一个高质量的分割掩码（segmentation mask）。这个方面被称为 Mask R-CNN，是在 Faster R-CNN 上的扩展——在其已有的用于边界框识别的分支上添加了一个并行的用于预测目标掩码的分支。

![img](https://img-blog.csdn.net/20180725102401977?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvcml6b25oZWFydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![img](https://img-blog.csdnimg.cn/20181220153956362.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NzEzODMx,size_16,color_FFFFFF,t_70)

- 首先，输入一幅你想处理的图片，然后进行对应的预处理操作，或者预处理后的图片；

- 然后，将其输入到一个预训练好的神经网络中（ResNet等）获得对应的feature map；

- 接着，对这个feature map中的每一点设定预定个的ROI，从而获得多个候选ROI；

- 接着，将这些候选的ROI送入RPN网络进行二值分类（前景或背景）和BB回归，过滤掉一部分候选的ROI；

- 接着，对这些剩下的ROI进行ROIAlign操作（即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来）；


- 最后，对这些ROI进行分类（N类别分类）、BB回归和MASK生成（在每一个ROI里面进行FCN操作）。

Mask R-CNN的训练损失函数可以描述为：

![img](https://img-blog.csdnimg.cn/20181220154046401.png)

 损失函数：分类误差+检测误差+分割误差，即L=Lcls+Lbox+Lmask 

 Lcls、Lbox：利用全连接预测出每个RoI的所属类别及其矩形框坐标值

 Lmask：

1. mask分支采用FCN对每个RoI的分割输出维数为$ K*m*m $（其中：m表示RoI Align特征图的大小），即K个类别的$m*m$的二值mask;保持$ m*m $的空间布局，pixel-to-pixel操作需要保证RoI特征 映射到原图的对齐性，这也是使用RoIAlign解决对齐问题原因，减少像素级别对齐的误差。
2. $ K*m*m $二值mask结构解释：最终的FCN输出一个K层的mask，每一层为一类，Log输出，用0.5作为阈值进行二值化，产生背景和前景的分割Mask
3. 这样，Lmask 使得网络能够输出每一类的 mask，且不会有不同类别 mask 间的竞争. 分类网络分支预测 object 类别标签，以选择输出 mask，对每一个ROI，如果检测得到ROI属于哪一个分 类，就只使用哪一个分支的相对熵误差作为误差值进行计算。（举例说明：分类有3类（猫，狗，人），检测得到当前ROI属于“人”这一类，那么所使用的Lmask为“人”这一分支的mask，即，每个class类别对应一个mask可以有效避免类间竞争（其他class不贡献Loss）
4. 对每一个像素应用sigmoid，然后取RoI上所有像素的交叉熵的平均值作为Lmask。

```python

```

## SSDVgg

![img](https://img-blog.csdn.net/20180305162531192)

![img](https://img-blog.csdn.net/20180305162545697)

### SSD算法步骤：

1. 输入一幅图片（200x200），将其输入到预训练好的分类网络中来获得不同大小的特征映射，修改了传统的VGG16网络；

- 将VGG16的FC6和FC7层转化为卷积层，如图1上的Conv6和Conv7；
- 去掉所有的Dropout层和FC8层；
- 添加了Atrous算法（hole算法）；
- 将Pool5从2x2-S2变换到3x3-S1；

2. 抽取Conv4_3、Conv7、Conv8_2、Conv9_2、Conv10_2、Conv11_2层的feature map，然后分别在这些feature map层上面的每一个点构造6个不同尺度大小的BB，然后分别进行检测和分类，生成多个BB，如图1下面的图所示；

3. 将不同feature map获得的BB结合起来，经过NMS（非极大值抑制）方法来抑制掉一部分重叠或者不正确的BB，生成最终的BB集合（即检测结果）；

### Defalut box生成规则

- 以feature map上每个点的中点为中心（offset=0.5），生成一系列同心的Defalut box（然后中心点的坐标会乘以step，相当于从feature map位置映射回原图位置）
- 使用m(SSD300中m=6)个不同大小的feature map 来做预测，最底层的 feature map 的 scale 值为 Smin=0.2，最高层的为Smax=0.95，其他层通过下面的公式计算得到：
- ![img](https://img-blog.csdn.net/20180305170118530)
- 使用不同的ratio值，[1, 2, 3, 1/2, 1/3]，通过下面的公式计算 default box 的宽度w和高度h
- ![img](https://img-blog.csdn.net/20180305170410217)
- 而对于ratio=0的情况，指定的scale如下所示，即总共有 6 中不同的 default box。

- ![img](https://img-blog.csdn.net/20180305170622682)

### LOSS计算

与常见的 Object Detection模型的目标函数相同，SSD算法的目标函数分为两部分：计算相应的default box与目标类别的confidence loss以及相应的位置回归。

![img](https://img-blog.csdn.net/20180305172222601)

其中N是match到Ground Truth的default box数量；而alpha参数用于调整confidence loss和location loss之间的比例，默认alpha=1。

位置回归则是采用 Smooth L1 loss，目标函数为:

![img](https://img-blog.csdn.net/20180305172811596)

confidence loss是典型的softmax loss：

![img](https://img-blog.csdn.net/20180305172942971)

### Atrous Algothrim（获得更加密集的得分映射）

![img](https://img-blog.csdn.net/20180305182931114)

作用：既想利用已经训练好的模型进行fine-tuning，又想改变网络结构得到更加dense的score map。

![img](https://bbsmax.ikafan.com/static/L3Byb3h5L2h0dHAvaW1hZ2VzMjAxNS5jbmJsb2dzLmNvbS9ibG9nLzEwNTgyNjgvMjAxNjEyLzEwNTgyNjgtMjAxNjEyMjExMjE4MTA5MzItMTk5NDAwNjE5OC5qcGc=.jpg)

### NMS非极大值抑制

NMS 获取按照分数排序的建议列表并对已排序的列表进行迭代，丢弃那些 IoU 值大于某个预定义阈值的建议，并提出一个具有更高分数的建议。总之，抑制的过程是一个迭代-遍历-消除的过程。

```python
"""
SSD net (vgg_based) 300x300
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf

from ssd_layers import conv2d, max_pool2d, l2norm, dropout, \
    pad2d, ssd_multibox_layer
from ssd_anchors import ssd_anchors_all_layers

# SSD parameters
SSDParams = namedtuple('SSDParameters', ['img_shape',  # the input image size: 300x300
                                         'num_classes',  # number of classes: 20+1
                                         'no_annotation_label',
                                         'feat_layers', # list of names of layer for detection
                                         'feat_shapes', # list of feature map sizes of layer for detection
                                         'anchor_size_bounds', # the down and upper bounds of anchor sizes
                                         'anchor_sizes',   # list of anchor sizes of layer for detection
                                         'anchor_ratios',  # list of rations used in layer for detection
                                         'anchor_steps',   # list of cell size (pixel size) of layer for detection
                                         'anchor_offset',  # the center point offset
                                         'normalizations', # list of normalizations of layer for detection
                                         'prior_scaling'   #
                                         ])
class SSD(object):
    """SSD net 300"""
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=21,
                                    no_annotation_label=21,
                                    feat_layers=["block4", "block7", "block8", "block9", "block10", "block11"],
                                    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                    anchor_size_bounds=[0.15, 0.90],  # diff from the original paper
                                    anchor_sizes=[(21., 45.),
                                                  (45., 99.),
                                                  (99., 153.),
                                                  (153., 207.),
                                                  (207., 261.),
                                                  (261., 315.)],
                                    anchor_ratios=[[2, .5],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5],
                                                   [2, .5]],
                                    anchor_steps=[8, 16, 32, 64, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[20, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2]
                                    )

        predictions, logits, locations = self._built_net()
        #self._update_feat_shapes_from_net()
        classes, scores, bboxes = self._bboxes_select(predictions, locations)
        self._classes = classes
        self._scores = scores
        self._bboxes = bboxes

    def _built_net(self):
        """Construct the SSD net"""
        self.end_points = {}  # record the detection layers output
        self._images = tf.placeholder(tf.float32, shape=[None, self.ssd_params.img_shape[0],
                                                        self.ssd_params.img_shape[1], 3])
        with tf.variable_scope("ssd_300_vgg"):
            # original vgg layers
            # block 1
            net = conv2d(self._images, 64, 3, scope="conv1_1")
            net = conv2d(net, 64, 3, scope="conv1_2")
            self.end_points["block1"] = net
            net = max_pool2d(net, 2, scope="pool1")
            # block 2
            net = conv2d(net, 128, 3, scope="conv2_1")
            net = conv2d(net, 128, 3, scope="conv2_2")
            self.end_points["block2"] = net
            net = max_pool2d(net, 2, scope="pool2")
            # block 3
            net = conv2d(net, 256, 3, scope="conv3_1")
            net = conv2d(net, 256, 3, scope="conv3_2")
            net = conv2d(net, 256, 3, scope="conv3_3")
            self.end_points["block3"] = net
            net = max_pool2d(net, 2, scope="pool3")
            # block 4
            net = conv2d(net, 512, 3, scope="conv4_1")
            net = conv2d(net, 512, 3, scope="conv4_2")
            net = conv2d(net, 512, 3, scope="conv4_3")
            self.end_points["block4"] = net
            net = max_pool2d(net, 2, scope="pool4")
            # block 5
            net = conv2d(net, 512, 3, scope="conv5_1")
            net = conv2d(net, 512, 3, scope="conv5_2")
            net = conv2d(net, 512, 3, scope="conv5_3")
            self.end_points["block5"] = net
            print(net)
            net = max_pool2d(net, 3, stride=1, scope="pool5")
            print(net)

            # additional SSD layers
            # block 6: use dilate conv
            net = conv2d(net, 1024, 3, dilation_rate=6, scope="conv6")
            self.end_points["block6"] = net
            #net = dropout(net, is_training=self.is_training)
            # block 7
            net = conv2d(net, 1024, 1, scope="conv7")
            self.end_points["block7"] = net
            # block 8
            net = conv2d(net, 256, 1, scope="conv8_1x1")
            net = conv2d(pad2d(net, 1), 512, 3, stride=2, scope="conv8_3x3",
                         padding="valid")
            self.end_points["block8"] = net
            # block 9
            net = conv2d(net, 128, 1, scope="conv9_1x1")
            net = conv2d(pad2d(net, 1), 256, 3, stride=2, scope="conv9_3x3",
                         padding="valid")
            self.end_points["block9"] = net
            # block 10
            net = conv2d(net, 128, 1, scope="conv10_1x1")
            net = conv2d(net, 256, 3, scope="conv10_3x3", padding="valid")
            self.end_points["block10"] = net
            # block 11
            net = conv2d(net, 128, 1, scope="conv11_1x1")
            net = conv2d(net, 256, 3, scope="conv11_3x3", padding="valid")
            self.end_points["block11"] = net

            # class and location predictions
            predictions = []
            logits = []
            locations = []
            for i, layer in enumerate(self.ssd_params.feat_layers):
                cls, loc = ssd_multibox_layer(self.end_points[layer], self.ssd_params.num_classes,
                                              self.ssd_params.anchor_sizes[i],
                                              self.ssd_params.anchor_ratios[i],
                                              self.ssd_params.normalizations[i], scope=layer+"_box")
                predictions.append(tf.nn.softmax(cls))
                logits.append(cls)
                locations.append(loc)
            return predictions, logits, locations

    def _update_feat_shapes_from_net(self, predictions):
        """ Obtain the feature shapes from the prediction layers"""
        new_feat_shapes = []
        for l in predictions:
            new_feat_shapes.append(l.get_shape().as_list()[1:])
        self.ssd_params._replace(feat_shapes=new_feat_shapes)

    def anchors(self):
        """Get sSD anchors"""
        return ssd_anchors_all_layers(self.ssd_params.img_shape,
                                      self.ssd_params.feat_shapes,
                                      self.ssd_params.anchor_sizes,
                                      self.ssd_params.anchor_ratios,
                                      self.ssd_params.anchor_steps,
                                      self.ssd_params.anchor_offset,
                                      np.float32)

    def _bboxes_decode_layer(self, feat_locations, anchor_bboxes, prior_scaling):
        """
        Decode the feat location of one layer
        params:
         feat_locations: 5D Tensor, [batch_size, size, size, n_anchors, 4]
         anchor_bboxes: list of Tensors(y, x, w, h)
                        shape: [size,size,1], [size, size,1], [n_anchors], [n_anchors]
         prior_scaling: list of 4 floats
        """
        yref, xref, href, wref = anchor_bboxes
        print(yref)
        # Compute center, height and width
        cx = feat_locations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
        cy = feat_locations[:, :, :, :, 1] * href * prior_scaling[1] + yref
        w = wref * tf.exp(feat_locations[:, :, :, :, 2] * prior_scaling[2])
        h = href * tf.exp(feat_locations[:, :, :, :, 3] * prior_scaling[3])
        # compute boxes coordinates (ymin, xmin, ymax,,xmax)
        bboxes = tf.stack([cy - h / 2., cx - w / 2.,
                           cy + h / 2., cx + w / 2.], axis=-1)
        # shape [batch_size, size, size, n_anchors, 4]
        return bboxes

    def _bboxes_select_layer(self, feat_predictions, feat_locations, anchor_bboxes,
                             prior_scaling):
        """Select boxes from the feat layer, only for bacth_size=1"""
        n_bboxes = np.product(feat_predictions.get_shape().as_list()[1:-1])
        # decode the location
        bboxes = self._bboxes_decode_layer(feat_locations, anchor_bboxes, prior_scaling)
        bboxes = tf.reshape(bboxes, [n_bboxes, 4])
        predictions = tf.reshape(feat_predictions, [n_bboxes, self.ssd_params.num_classes])
        # remove the background predictions
        sub_predictions = predictions[:, 1:]
        # choose the max score class
        classes = tf.argmax(sub_predictions, axis=1) + 1  # class labels
        scores = tf.reduce_max(sub_predictions, axis=1)   # max_class scores
        # Boxes selection: use threshold
        filter_mask = scores > self.threshold
        classes = tf.boolean_mask(classes, filter_mask)
        scores = tf.boolean_mask(scores, filter_mask)
        bboxes = tf.boolean_mask(bboxes, filter_mask)
        return classes, scores, bboxes

    def _bboxes_select(self, predictions, locations):
        """Select all bboxes predictions, only for bacth_size=1"""
        anchor_bboxes_list = self.anchors()
        classes_list = []
        scores_list = []
        bboxes_list = []
        # select bboxes for each feat layer
        for n in range(len(predictions)):
            anchor_bboxes = list(map(tf.convert_to_tensor, anchor_bboxes_list[n]))
            classes, scores, bboxes = self._bboxes_select_layer(predictions[n],
                            locations[n], anchor_bboxes, self.ssd_params.prior_scaling)
            classes_list.append(classes)
            scores_list.append(scores)
            bboxes_list.append(bboxes)
        # combine all feat layers
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)
        return classes, scores, bboxes

    def images(self):
        return self._images

    def detections(self):
        return self._classes, self._scores, self._bboxes


if __name__ == "__main__":
    ssd = SSD()
    sess = tf.Session()
    saver_ = tf.train.Saver()
    saver_.restore(sess, "../ssd_vgg_300_weights.ckpt")


```

```python
"""
SSD anchors
"""
import math

import numpy as np

def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes

def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)  # [size, size, 1]
    x = np.expand_dims(x, axis=-1)  # [size, size, 1]

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    w = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors
```

```python
"""
Layers for SSD
"""

import tensorflow as tf

# Conv2d: for stride = 1
def conv2d(x, filters, kernel_size, stride=1, padding="same",
           dilation_rate=1, activation=tf.nn.relu, scope="conv2d"):
    kernel_sizes = [kernel_size] * 2
    strides = [stride] * 2
    dilation_rate = [dilation_rate] * 2
    return tf.layers.conv2d(x, filters, kernel_sizes, strides=strides,
                            dilation_rate=dilation_rate, padding=padding,
                            name=scope, activation=activation)

# max pool2d: default pool_size = stride
def max_pool2d(x, pool_size, stride=None, scope="max_pool2d"):
    pool_sizes = [pool_size] * 2
    strides = [pool_size] * 2 if stride is None else [stride] * 2
    return tf.layers.max_pooling2d(x, pool_sizes, strides, name=scope, padding="same")

# pad2d: for conv2d with stride > 1
def pad2d(x, pad):
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

# dropout
def dropout(x, rate=0.5, is_training=True):
    return tf.layers.dropout(x, rate=rate, training=is_training)

# l2norm (not bacth norm, spatial normalization)
def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma


# multibox layer: get class and location predicitions from detection layer
def ssd_multibox_layer(x, num_classes, sizes, ratios, normalization=-1, scope="multibox"):
    pre_shape = x.get_shape().as_list()[1:-1]
    pre_shape = [-1] + pre_shape
    with tf.variable_scope(scope):
        # l2 norm
        if normalization > 0:
            x = l2norm(x, normalization)
            print(x)
        # numbers of anchors
        n_anchors = len(sizes) + len(ratios)
        # location predictions
        loc_pred = conv2d(x, n_anchors*4, 3, activation=None, scope="conv_loc")
        loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])
        # class prediction
        cls_pred = conv2d(x, n_anchors*num_classes, 3, activation=None, scope="conv_cls")
        cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
        return cls_pred, loc_pred
```

```python
"""
Help functions for SSD
"""

import cv2
import numpy as np


############## preprocess image ##################
# whiten the image
def whiten_image(image, means=(123., 117., 104.)):
    """Subtracts the given means from each image channel"""
    if image.ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = np.array(means, dtype=image.dtype)
    image = image - mean
    return image

def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)

def preprocess_image(image):
    """Preprocess a image to inference"""
    image_cp = np.copy(image).astype(np.float32)
    # whiten the image
    image_whitened = whiten_image(image_cp)
    # resize the image
    image_resized = resize_image(image_whitened)
    # expand the batch_size dim
    image_expanded = np.expand_dims(image_resized, axis=0)
    return image_expanded

############## process bboxes ##################
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes

def process_bboxes(rclasses, rscores, rbboxes, rbbox_img = (0.0, 0.0, 1.0, 1.0),
                   top_k=400, nms_threshold=0.5):
    """Process the bboxes including sort and nms"""
    rbboxes = bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k)
    rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold)
    rbboxes = bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
```

```python

# ==============================================================================
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm


# class names
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5, show_class_name=True):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = CLASSES[cls_id-1] if show_class_name else str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()
```

## yolo

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180130221342432?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2h1MjAyMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

损失函数：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180130221734615?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2h1MjAyMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```python
"""
Yolo V1 by tensorflow
"""

import numpy as np
import tensorflow as tf
import cv2


def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

class Yolo(object):
    def __init__(self, weights_file, verbose=True):
        self.verbose = verbose
        # detection params
        self.S = 7  # cell size
        self.B = 2  # boxes_per_cell
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
        self.C = len(self.classes) # number of classes
        # offset for box center (top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                              [self.B, self.S, self.S]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.threshold = 0.2  # confidence scores threhold
        self.iou_threshold = 0.4
        #  the maximum number of boxes to be selected by non max suppression
        self.max_output_size = 10

        self.sess = tf.Session()
        self._build_net()
        self._build_detector()
        self._load_weights(weights_file)

    def _build_net(self):
        """build the network"""
        if self.verbose:
            print("Start to build the network ...")
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])
        net = self._conv_layer(self.images, 1, 64, 7, 2)
        net = self._maxpool_layer(net, 1, 2, 2)
        net = self._conv_layer(net, 2, 192, 3, 1)
        net = self._maxpool_layer(net, 2, 2, 2)
        net = self._conv_layer(net, 3, 128, 1, 1)
        net = self._conv_layer(net, 4, 256, 3, 1)
        net = self._conv_layer(net, 5, 256, 1, 1)
        net = self._conv_layer(net, 6, 512, 3, 1)
        net = self._maxpool_layer(net, 6, 2, 2)
        net = self._conv_layer(net, 7, 256, 1, 1)
        net = self._conv_layer(net, 8, 512, 3, 1)
        net = self._conv_layer(net, 9, 256, 1, 1)
        net = self._conv_layer(net, 10, 512, 3, 1)
        net = self._conv_layer(net, 11, 256, 1, 1)
        net = self._conv_layer(net, 12, 512, 3, 1)
        net = self._conv_layer(net, 13, 256, 1, 1)
        net = self._conv_layer(net, 14, 512, 3, 1)
        net = self._conv_layer(net, 15, 512, 1, 1)
        net = self._conv_layer(net, 16, 1024, 3, 1)
        net = self._maxpool_layer(net, 16, 2, 2)
        net = self._conv_layer(net, 17, 512, 1, 1)
        net = self._conv_layer(net, 18, 1024, 3, 1)
        net = self._conv_layer(net, 19, 512, 1, 1)
        net = self._conv_layer(net, 20, 1024, 3, 1)
        net = self._conv_layer(net, 21, 1024, 3, 1)
        net = self._conv_layer(net, 22, 1024, 3, 2)
        net = self._conv_layer(net, 23, 1024, 3, 1)
        net = self._conv_layer(net, 24, 1024, 3, 1)
        net = self._flatten(net)
        net = self._fc_layer(net, 25, 512, activation=leak_relu)
        net = self._fc_layer(net, 26, 4096, activation=leak_relu)
        net = self._fc_layer(net, 27, self.S*self.S*(self.C+5*self.B))
        self.predicts = net

    def _build_detector(self):
        """Interpret the net output and get the predicted boxes"""
        # the width and height of orignal image
        self.width = tf.placeholder(tf.float32, name="img_w")
        self.height = tf.placeholder(tf.float32, name="img_h")
        # get class prob, confidence, boxes from net output
        idx1 = self.S * self.S * self.C
        idx2 = idx1 + self.S * self.S * self.B
        # class prediction
        class_probs = tf.reshape(self.predicts[0, :idx1], [self.S, self.S, self.C])
        # confidence
        confs = tf.reshape(self.predicts[0, idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x, y, w, h)
        boxes = tf.reshape(self.predicts[0, idx2:], [self.S, self.S, self.B, 4])

        # convert the x, y to the coordinates relative to the top left point of the image
        # the predictions of w, h are the square root
        # multiply the width and height of image
        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S * self.width,
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S * self.height,
                          tf.square(boxes[:, :, :, 2]) * self.width,
                          tf.square(boxes[:, :, :, 3]) * self.height], axis=3)

        # class-specific confidence scores [S, S, B, C]
        scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)

        scores = tf.reshape(scores, [-1, self.C])  # [S*S*B, C]
        boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]

        # find each box class, only select the max score
        box_classes = tf.argmax(scores, axis=1)
        box_class_scores = tf.reduce_max(scores, axis=1)

        # filter the boxes by the score threshold
        filter_mask = box_class_scores >= self.threshold
        scores = tf.boolean_mask(box_class_scores, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)
        box_classes = tf.boolean_mask(box_classes, filter_mask)

        # non max suppression (do not distinguish different classes)
        # ref: https://tensorflow.google.cn/api_docs/python/tf/image/non_max_suppression
        # box (x, y, w, h) -> box (x1, y1, x2, y2)
        _boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                           boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
        nms_indices = tf.image.non_max_suppression(_boxes, scores,
                                                   self.max_output_size, self.iou_threshold)
        self.scores = tf.gather(scores, nms_indices)
        self.boxes = tf.gather(boxes, nms_indices)
        self.box_classes = tf.gather(box_classes, nms_indices)

    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        """Conv layer"""
        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters,]))
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
        output = leak_relu(tf.nn.bias_add(conv, bias))
        if self.verbose:
            print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s" \
                  % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out,]))
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("    Layer %d: type=Fc, num_out=%d, output_shape=%s" \
                  % (id, num_out, str(output.get_shape())))
        return output

    def _maxpool_layer(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")
        if self.verbose:
            print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s" \
                  % (id, pool_size, stride, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])

    def _load_weights(self, weights_file):
        """Load weights from file"""
        if self.verbose:
            print("Start to load weights from file:%s" % (weights_file))
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file)

    def detect_from_file(self, image_file, imshow=True, deteted_boxes_file="boxes.txt",
                     detected_image_file="detected_image.jpg"):
        """Do detection given a image file"""
        # read image
        image = cv2.imread(image_file)
        img_h, img_w, _ = image.shape
        scores, boxes, box_classes = self._detect_from_image(image)
        predict_boxes = []
        for i in range(len(scores)):
            predict_boxes.append((self.classes[box_classes[i]], boxes[i, 0],
                                boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
        self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)

    def _detect_from_image(self, image):
        """Do detection given a cv image"""
        img_h, img_w, _ = image.shape
        img_resized = cv2.resize(image, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
                    feed_dict={self.images: _images, self.width: img_w, self.height: img_h})
        return scores, boxes, box_classes

    def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
                     detected_image_file=None):
        """Show the detection boxes"""
        img_cp = image.copy()
        if deteted_boxes_file:
            f = open(deteted_boxes_file, "w")
        #  draw boxes
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.verbose:
                print("   class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (results[i][0],
                            x, y, w, h, results[i][-1]))

                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if deteted_boxes_file:
                f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
                        str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(1)
        if detected_image_file:
            cv2.imwrite(detected_image_file, img_cp)
        if deteted_boxes_file:
            f.close()

if __name__ == "__main__":
    yolo_net = Yolo("./YOLO_small.ckpt")
    yolo_net.detect_from_file("./test/3.jpg")
```


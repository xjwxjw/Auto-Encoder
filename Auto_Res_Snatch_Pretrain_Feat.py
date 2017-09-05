import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.misc as misc
from tensorflow.contrib import rnn
import os
def conv(batch_input, out_channels, stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conved

def lrelu(x, a,scope_name):
    with tf.name_scope("lrelu-" + scope_name):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input,scope_name, trainable = True):
    with tf.variable_scope("batchnorm-" + scope_name):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def deconv(batch_input, out_channels,scope_name):
    with tf.variable_scope("deconv-" + scope_name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        deconved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return deconved

def encoder_layer(input, leaky_rate = 0.2, out_channels = 64, stride = 2, scope_name = 'encoder_layer', trainable = True):
    rectified = lrelu(input, 0.2,scope_name = scope_name)
    convolved = conv(rectified, out_channels, stride=2,scope_name = scope_name)
    #output = batchnorm(convolved, scope_name = scope_name)
    #output = tf.contrib.layers.batch_norm(convolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return convolved

def decoder_layer(input, out_channels = 64, stride = 2, scope_name = 'decoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    deconvolved = deconv(rectified, out_channels, scope_name = scope_name)
    #output = batchnorm(deconvolved, scope_name = scope_name, trainable = trainable)
    #output = tf.contrib.layers.batch_norm(deconvolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return deconvolved
start_order = [1,21,12,36,1,12,7,1,17,1,1,27,27,47,16,9,8,6,1]
dir_name = ['03_01','03_02','03_03','03_04','03_05','03_06','07_01','07_02','07_03','07_04','07_05','11_01','11_02','11_03','11_04','18_01','18_02','18_03','18_04']
class AutoEncoder(object):
    def __init__(self, trainable):
        self.batch_size = 32
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.NUM_CHANNELS = 3
        self.iterations = 500000
        self.lamda_recon = 1
        self.lamda_appr = 32
        self.lamda_pose = 32
        self.logs_dir = "./Naive4_"+str(self.lamda_recon)+"_"+str(self.lamda_appr)+"_"+str(self.lamda_pose)+"_Result_Normalize_Feat_Sigmoid"
        self.images_dir = "/home/xjwxjw/Documents/DualSpaceTranformation/SnatchDataset/Naive4/"
        self.learning_rate = 1e-4
        self.trainable = trainable
        self.rnn_unit = 1024
        self.time_step = 64
        self.dir_list = []
        self.dir_start = []
        if not os.path.exists(os.path.join(self.logs_dir)):
            os.mkdir(os.path.join(self.logs_dir))
        if not os.path.exists(os.path.join(self.logs_dir,'train')):
            os.mkdir(os.path.join(self.logs_dir,'train'))
        if not os.path.exists(os.path.join(self.logs_dir,'val')):
            os.mkdir(os.path.join(self.logs_dir,'val'))
        #for line in open('filtered_sequence.txt','r'):
            #self.dir_list.append(line.split('\n')[0].split(' ')[0])
            #self.dir_start.append(line.split('\n')[0].split(' ')[1])
        #self.dir_num = len(self.dir_list)
    def _LSTM(self, input_hidden_state, scope_name = "lstm", trainable = True):
        with tf.variable_scope(scope_name) as scope:
            print input_hidden_state.get_shape()
            w_in = tf.get_variable("w_in", [input_hidden_state.get_shape()[1],self.rnn_unit], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
            w_out = tf.get_variable("w_out", [self.rnn_unit,input_hidden_state.get_shape()[1]], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
            b_in = tf.get_variable("b_in", [self.rnn_unit,], dtype=tf.float32, initializer=tf.zeros_initializer())
            b_out = tf.get_variable("b_out", [input_hidden_state.get_shape()[1],], dtype=tf.float32, initializer=tf.zeros_initializer())  
            input_rnn=tf.matmul(input_hidden_state,w_in)+b_in
            input_rnn=tf.reshape(input_rnn,[-1,self.time_step,self.rnn_unit]) 
            cell=rnn.BasicLSTMCell(self.rnn_unit)
            init_state=cell.zero_state(1,dtype=tf.float32)
            output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32) 
            tf.summary.histogram(output_rnn.op.name + "/activation", output_rnn)
            output=tf.reshape(output_rnn,[-1,self.rnn_unit])
            pred=tf.matmul(output,w_out)+b_out          #==>(32,512)
            pred = tf.expand_dims(pred, 1)              #==>(32,1,512)
            pred = tf.expand_dims(pred, 1)              #==>(32,1,1,512)
            tf.summary.histogram(pred.op.name + "/activation", pred)
        return pred
    def _encoder(self, input_images, scope_name = "encoder", trainable = True, scope_reuse = False):
        hidden_state = []
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print input_images.get_shape()
            output = conv(input_images, 32, stride=2 ,scope_name = 'encoder_layer1')#(256, 256, 3)==>(128, 128, 32)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 64, stride = 2, scope_name = 'encoder_layer2', trainable = trainable)#(128, 128, 32)==>(64, 64, 64)
            print output.get_shape()   
            tf.summary.histogram(output.op.name + "/activation", output)
         
            hidden_state = encoder_layer(output, leaky_rate = 0.2, out_channels = 128, stride = 2, scope_name = 'encoder_layer3', trainable = trainable)#(64, 64, 64)==>(32, 32, 128)
            hidden_state = tf.sigmoid(hidden_state)
            print hidden_state.get_shape()
            tf.summary.histogram(hidden_state.op.name + "/activation", output)

            #output = encoder_layer(output, leaky_rate = 0.2, out_channels = 64, stride = 2, scope_name = 'encoder_layer4', trainable = trainable)#(32, 32, 32)==>(16, 16, 64)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = encoder_layer(output, leaky_rate = 0.2, out_channels = 128, stride = 2, scope_name = 'encoder_layer5', trainable = trainable)#(16, 16, 64)==>(8, 8, 128)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = encoder_layer(output, leaky_rate = 0.2, out_channels = 256, stride = 2, scope_name = 'encoder_layer6', trainable = trainable)#(8, 8, 128)==>(4, 4, 256)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)
 
            #output = encoder_layer(output, leaky_rate = 0.2, out_channels = 512, stride = 2, scope_name = 'encoder_layer7', trainable = trainable)#(4, 4, 256)==>(2, 2, 512)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)
 
            #hidden_state = encoder_layer(output, leaky_rate = 0.2, out_channels = 512, stride = 2, scope_name = 'encoder_layer8', trainable = trainable)#(2, 2, 512)==>(1, 1, 512)
            #print hidden_state.get_shape()
            #tf.summary.histogram(hidden_state.op.name + "/activation", hidden_state)
        return hidden_state
    def _decoder(self, input_hidden_state, scope_name = "decoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            #output = decoder_layer(input_hidden_state, out_channels = 256, stride = 4, scope_name = 'decoder_layer1', trainable = trainable)#(1, 1, 512)==>(2, 2, 256)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = decoder_layer(output, out_channels = 256, stride = 2, scope_name = 'decoder_layer2', trainable = trainable)#(2, 2, 256)==>(4, 4, 256)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = decoder_layer(output, out_channels = 256, stride = 2, scope_name = 'decoder_layer3', trainable = trainable)#(4, 4, 256)==>(8, 8, 256)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = decoder_layer(output, out_channels = 128, stride = 2, scope_name = 'decoder_layer4', trainable = trainable)#(8, 8, 256)==>(16, 16, 128)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            #output = decoder_layer(output, out_channels = 64, stride = 2, scope_name = 'decoder_layer5', trainable = trainable)#(16, 16, 128)==>(32, 32, 64)
            #print output.get_shape()
            #tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(input_hidden_state, out_channels = 64, stride = 2, scope_name = 'decoder_layer6', trainable = trainable)#(32, 32, 128)==>(64, 64, 64)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 32, stride = 2, scope_name = 'decoder_layer7', trainable = trainable)#(64, 64, 32)==>(128, 128, 16)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            generated_images = decoder_layer(output, out_channels = 3, stride = 2, scope_name = 'decoder_layer8', trainable = trainable)#(128, 128, 16)==>(256, 256, 3)
            print generated_images.get_shape()
            tf.summary.histogram(generated_images.op.name + "/activation", generated_images)
        return generated_images
    def construct_network(self):
        self.input_image = tf.placeholder(tf.float32, [self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,3], name="input_imager1")
        self.hidden_state = self._encoder(self.input_image,trainable = self.trainable) 
        self.generated_images = self._decoder(self.hidden_state, trainable = self.trainable)
        self.train_variables = tf.trainable_variables()
        self.recon_loss = self.lamda_recon * (tf.reduce_sum(tf.squared_difference(self.input_image,self.generated_images)))
        self.recon_variables = [v for v in self.train_variables]
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1 = 0.9)
        self.recon_grads = self.optimizer.compute_gradients(self.recon_loss, var_list=self.recon_variables)
        self.recon_op = self.optimizer.apply_gradients(self.recon_grads)
        tf.summary.scalar("recon_loss", self.recon_loss)
    def _save_images(self,input_images,prefix):
        for i in range(self.batch_size):
            input_images[i] *= 127.5
            input_images[i] += 127.5
            input_images[i] = np.clip(input_images[i], 0, 255).astype(np.uint8)
            input_images[i] = np.reshape(input_images[i], (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, -1))
            misc.imsave(self.logs_dir+prefix+"_"+str(i)+".jpg", input_images[i])
    def train(self):
        self.config = tf.ConfigProto()  
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.config)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(self.iterations):
            cur_dir1 = np.random.choice(1900)
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.images_dir+str(int(cur_dir1/19)+1)+'/'+str(dir_name[cur_dir1%19])+'/'+str(i+start_order[cur_dir1%19])+'.jpg'))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            _, recon_loss_eval = self.sess.run([self.recon_op,self.recon_loss],feed_dict={self.input_image:img_batch})
            print str(itr)+'recon: '+str(recon_loss_eval)
            if itr % 100 == 99:
                hidden_state_eval,gen_images_eval = self.sess.run([self.hidden_state,self.generated_images],feed_dict={self.input_image:img_batch})
                sio.savemat(self.logs_dir+'/feature_train.mat', {'feature_train': hidden_state_eval})
                self._save_images(img_batch,"/train/ori1_"+str(0))
                self._save_images(gen_images_eval,"/train/gen1_"+str(0))
                cur_dir1 = np.random.choice(380)
                for i in range(self.batch_size):
                    cur_img = np.float32(misc.imread(self.images_dir+'Val/'+str(int(cur_dir1/19)+1)+'/'+str(dir_name[cur_dir1%19])+'/'+str(i+start_order[cur_dir1%19])+'.jpg'))
                    cur_img -= 127.5
                    cur_img /= 127.5
                    img_batch[i] = cur_img
                hidden_state_eval,gen_images_eval = self.sess.run([self.hidden_state,self.generated_images],feed_dict={self.input_image:img_batch})
                self._save_images(img_batch,"/val/ori1_"+str(0))
                self._save_images(gen_images_eval,"/val/gen1_"+str(0))
                sio.savemat(self.logs_dir+'/feature_val.mat', {'feature_val': hidden_state_eval})
            if itr % 2000 == 99:
                self.saver.save(self.sess, self.logs_dir + "/model.ckpt", global_step=itr)
        coord.request_stop()
        coord.join(threads)
    def test(self):
        self.config = tf.ConfigProto()  
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.config)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #saver_params={}
        #reader = pywrap_tensorflow.NewCheckpointReader('/home/xjwxjw/Documents/DualSpaceTranformation/Test/ResNet_Pretrain/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt')  
        #var_to_shape_map = reader.get_variable_to_shape_map()  
        #checkpoint_keys=var_to_shape_map.keys()
        #for v in params:
        #    v_name=v.name.split(':')[0]
        #    if v_name in checkpoint_keys:
        #        saver_params[v_name] = v
        #        print 'rec params: ',v_name
        #saver_res=tf.train.Saver(saver_params)
        #saver_res.restore(self.sess,'/home/xjwxjw/Documents/DualSpaceTranformation/Test/ResNet_Pretrain/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt')

        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(1900):
            print itr
            cur_dir1 = itr#np.random.choice(1900)
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.images_dir+str(int(cur_dir1/19)+1)+'/'+str(dir_name[cur_dir1%19])+'/'+str(i+start_order[cur_dir1%19])+'.jpg'))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            hidden_state_eval = self.sess.run(self.hidden_state,feed_dict={self.input_image:img_batch})
            #self._save_images(img_batch,"/val/ori1_"+str(itr))
            #self._save_images(gen_images_eval,"/val/gen1_"+str(itr))
            sio.savemat('/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/DataSet/Snatch_Hidden_State_128/feature_'+str(itr)+'.mat', {'feature_val': hidden_state_eval})
        coord.request_stop()
        coord.join(threads)

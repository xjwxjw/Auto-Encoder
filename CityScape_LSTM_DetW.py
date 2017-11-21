import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python import pywrap_tensorflow
import scipy.io as sio
import scipy.misc as misc
import os
from cell import ConvLSTMCell
from myLSTMLayer import MyConvLSTMCell

def conv(batch_input, out_channels, stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conved

def conv_detw(batch_input, out_channels, stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, 1, stride, 1], padding="VALID")
        return conved

def encoder_layer(input, leaky_rate = 0.2, out_channels = 64, stride = 2, scope_name = 'encoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    convolved = conv(rectified, out_channels, stride=stride,scope_name = scope_name)
    #output = batchnorm(convolved, scope_name = scope_name)
    #output = tf.contrib.layers.batch_norm(convolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return convolved

def deconv(batch_input, out_channels,stride,scope_name):
    with tf.variable_scope("deconv-" + scope_name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        deconved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        return deconved

def decoder_layer(input, out_channels = 64, stride = 2, scope_name = 'decoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    deconvolved = deconv(rectified, out_channels, stride = stride,scope_name = scope_name)
    return deconvolved

start_order = [1,21,12,36,1,12,7,1,17,1,1,27,27,47,16,9,8,6,1]
dir_name = ['03_01','03_02','03_03','03_04','03_05','03_06','07_01','07_02','07_03','07_04','07_05','11_01','11_02','11_03','11_04','18_01','18_02','18_03','18_04']
class AutoEncoder(object):
    def __init__(self, trainable):
        self.batch_size = 30
        self.IMAGE_HEIGHT = 128
        self.IMAGE_WIDTH = 256
        self.NUM_CHANNELS = 3
        self.iterations = 500000
        self.lamda_dis = 0
        self.lamda_reg = 100
        self.learning_rate = 1e-4
        self.logs_dir = "/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/Model/CityScape_RGB_DetW_LSTM2"+str(self.lamda_reg)+'_'+str(self.lamda_dis)+'_'+str(self.learning_rate)
        self.hidden_state_dir = "/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/DataSet/Naive5_Hidden_State_128/"
        self.images_dir = "/home/xjwxjw/Documents/DualSpaceTranformation/SnatchDataset/Naive4/"
        self.trainable = trainable
        self.rnn_unit = 1024
        self.time_step = 30
        self.conLSTM_kernel = [4,4]
        self.dir_list = []
        self.dir_start = []
        self.lstm_channel = 32
        self.imglist = []
        for line in open('RGBImgList.txt','r'):
            self.imglist.append(line.split('\n')[0].replace('_edge_ds1','_leftImg8bit'))
        if not os.path.exists(os.path.join(self.logs_dir)):
            os.mkdir(os.path.join(self.logs_dir))
        if not os.path.exists(os.path.join(self.logs_dir,'train')):
            os.mkdir(os.path.join(self.logs_dir,'train'))
        if not os.path.exists(os.path.join(self.logs_dir,'val')):
            os.mkdir(os.path.join(self.logs_dir,'val'))
    def _convLSTMDetW(self,input_hidden_state, W_DetW = None, scope_name = 'convLSTM', initial_state=None,trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print scope_name,input_hidden_state.get_shape()
            cell = MyConvLSTMCell([self.IMAGE_HEIGHT/8,self.IMAGE_WIDTH/8], self.lstm_channel, self.conLSTM_kernel)
            outputs, state = cell.run_rnn(x = input_hidden_state, W = W_DetW, state = initial_state)
            print scope_name,outputs.get_shape()
            return outputs, state
    def _convLSTM(self,input_hidden_state, scope_name = 'convLSTM', initial_state=None,trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            # Create a placeholder for videos.
            print scope_name,input_hidden_state.get_shape()
            cell = ConvLSTMCell([self.IMAGE_HEIGHT/8,self.IMAGE_WIDTH/8], self.lstm_channel, self.conLSTM_kernel)
            if initial_state == None:
                 outputs, state = tf.nn.dynamic_rnn(cell, input_hidden_state, initial_state=cell.zero_state(1,dtype=tf.float32),dtype=input_hidden_state.dtype)
            else:
                outputs, state = tf.nn.dynamic_rnn(cell, input_hidden_state, initial_state=initial_state,dtype=input_hidden_state.dtype)
            print scope_name,outputs.get_shape()
            return outputs, state, cell.return_weight()
    def _encode_lstm_hidden(self,input_hidden_state, scope_name = 'encode_lstm', initial_state=None,trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print 'encode_lstm_hidden_in',input_hidden_state.get_shape()
            output = encoder_layer(input_hidden_state, leaky_rate = 0.2, out_channels = 32, stride = 1, scope_name = 'encoder_lstm', trainable = trainable)#(32, 32, 128)==>(32, 32, 32)
            print 'encode_lstm_hidden_out',output.get_shape()   
            return output
    def _encoder(self, input_images, scope_name = "encoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print input_images.get_shape()
            output = conv(input_images, 8, stride=2 ,scope_name = 'encoder_layer1')#(256, 256, 3)==>(128, 128, 8)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 16, stride = 2, scope_name = 'encoder_layer2', trainable = trainable)#(128, 128, 8)==>(64, 64, 16)
            print output.get_shape()   
            tf.summary.histogram(output.op.name + "/activation", output)
         
            hidden_state = encoder_layer(output, leaky_rate = 0.2, out_channels = 32, stride = 2, scope_name = 'encoder_layer3', trainable = trainable)#(64, 64, 16)==>(32, 32, 32)
            hidden_state = tf.sigmoid(hidden_state)
            print hidden_state.get_shape()
            tf.summary.histogram(hidden_state.op.name + "/activation", output)
        return hidden_state
    def _encoder_detw(self, input_images, scope_name = "encoder_detw", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print input_images.get_shape()
            output = conv(input_images, 32, stride=2 ,scope_name = 'detw_layer1')#(32, 32, 32)==>(16, 16, 32)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = conv(output, out_channels = 32, stride = 2, scope_name = 'detw_layer2', )#(16, 16, 32)==>(8, 8, 32)
            print output.get_shape()   
            tf.summary.histogram(output.op.name + "/activation", output)
         
            output = conv_detw(output, out_channels = 32, stride = 2, scope_name = 'detw_layer3', )#(8, 8, 32)==>(4, 4, 32)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)
        return output
    def _decoder(self, input_hidden_state, scope_name = "decoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            output = decoder_layer(input_hidden_state, out_channels = 16, stride = 2, scope_name = 'decoder_layer6', trainable = trainable)#(32, 32, 32)==>(64, 64, 16)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 8, stride = 2, scope_name = 'decoder_layer7', trainable = trainable)#(64, 64, 16)==>(128, 128, 8)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            generated_images = decoder_layer(output, out_channels = 3, stride = 2, scope_name = 'decoder_layer8', trainable = trainable)#(128, 128, 8)==>(256, 256, 3)
            print generated_images.get_shape()
            tf.summary.histogram(generated_images.op.name + "/activation", generated_images)
            return generated_images
    def _discriminator(self,input_hidden_state,scope_name = "discriminator",trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            output = encoder_layer(input_hidden_state, out_channels = 32, stride = 1, scope_name = 'discriminator_layer1', trainable = trainable)#(32, 32, 128)==>(32, 32, 32)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, out_channels = 8, stride = 1, scope_name = 'discriminator_layer2', trainable = trainable)#(32, 32, 32)==>(32, 32, 8)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            dis_features = encoder_layer(output, out_channels = 3, stride = 1, scope_name = 'discriminator_layer3', trainable = trainable)#(32, 32, 8)==>(32, 32, 3)
            print dis_features.get_shape()
            tf.summary.histogram(dis_features.op.name + "/activation", dis_features)
            return dis_features 
    def construct_network(self):
        self.input_images = tf.placeholder(tf.float32, [self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,3], name="input_images")
        self.input_state = tf.expand_dims(self._encoder(self.input_images),0)
        self.encode_input_state,self.encode_hidden_state,_ = self._convLSTM(tf.expand_dims(tf.expand_dims(self.input_state[0,0],axis=0),axis=0),scope_name = 'ConvLSTMEncode')
        self.final_input_state = self.encode_input_state
        (self.c,self.h) = self.encode_hidden_state
        print 'output',self.final_input_state.get_shape()
        print 'c',self.c.get_shape()
        print 'h',self.h.get_shape()
        for i in range(14):
            self.encode_input_state,self.encode_hidden_state,_ = self._convLSTM(tf.expand_dims(tf.expand_dims(self.input_state[0,i+1],axis=0),axis=0),initial_state=self.encode_hidden_state,scope_name = 'ConvLSTMEncode',scope_reuse=True)
            (cur_c,cur_h) = self.encode_hidden_state
            self.c = tf.concat([self.c,cur_c],axis=3)
            self.h = tf.concat([self.h,cur_h],axis=3)
            self.final_input_state = tf.concat([self.final_input_state,self.encode_input_state],axis=1)
            print 'output',self.final_input_state.get_shape()
            print 'c',self.c.get_shape()
            print 'h',self.h.get_shape()

        print 'last4c',self.c[0,:,:,-4 * self.lstm_channel:].get_shape()
        print 'last4h',self.h[0,:,:,-4 * self.lstm_channel:].get_shape()
        self.last4_c = self._encode_lstm_hidden(tf.expand_dims(self.c[0,:,:,-4 * self.lstm_channel:],axis=0), scope_name = 'encode_lstm_c')
        self.last4_h = self._encode_lstm_hidden(tf.expand_dims(self.h[0,:,:,-4 * self.lstm_channel:],axis=0), scope_name = 'encode_lstm_h')
        self.last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(self.last4_c,self.last4_h)
        self.decode_input_state = tf.squeeze(self.encode_input_state,axis=0)

        self.W_t = tf.get_variable('kernel', self.conLSTM_kernel + [32*2, 4 * self.lstm_channel])     
        self.detW = tf.stack(tf.unstack(self._encoder_detw(tf.stack([-self.input_state[0,14,] + tf.expand_dims(self.input_state[0,15,:,:,i],-1) for i in range(32)],0)),axis=0),-1) 
        self.detW = tf.concat([self.detW,self.detW],-2)
        self.detW = tf.concat([self.detW,self.detW,self.detW,self.detW],-1)  
        self.W_t = self.W_t + self.detW
        self.decode_input_state,self.decode_hidden_state = self._convLSTMDetW(self.decode_input_state, W_DetW = self.W_t, initial_state=self.last4_hidden_state,scope_name = 'ConvLSTMDecode')
        (cur_c,cur_h) = self.decode_hidden_state
        self.c = tf.concat([self.c,cur_c],axis=3)
        self.h = tf.concat([self.h,cur_h],axis=3)
        self.final_input_state = tf.concat([self.final_input_state,tf.expand_dims(self.decode_input_state,axis=0)],axis=1)
        print 'output',self.final_input_state.get_shape()
        print 'c',self.c.get_shape()
        print 'h',self.h.get_shape()
        print self.final_input_state.get_shape()
        for i in range(14):
            self.last4_c = self._encode_lstm_hidden(tf.expand_dims(self.c[0,:,:,-4 * self.lstm_channel:],axis=0), scope_name = 'encode_lstm_c',scope_reuse=True)
            self.last4_h = self._encode_lstm_hidden(tf.expand_dims(self.h[0,:,:,-4 * self.lstm_channel:],axis=0), scope_name = 'encode_lstm_h',scope_reuse=True)
            self.last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(self.last4_c,self.last4_h)
            self.detW = tf.stack(tf.unstack(self._encoder_detw(tf.stack([-self.input_state[0,15+i,] + tf.expand_dims(self.input_state[0,16+i,:,:,j],-1) for j in range(32)],0),scope_reuse=True),axis=0),-1) 
            self.detW = tf.concat([self.detW,self.detW],-2)
            self.detW = tf.concat([self.detW,self.detW,self.detW,self.detW],-1)
            self.W_t = self.W_t + self.detW
            self.decode_input_state,self.decode_hidden_state = self._convLSTMDetW(self.decode_input_state, W_DetW = self.W_t, initial_state=self.last4_hidden_state,scope_name = 'ConvLSTMDecode',scope_reuse=True)
            (cur_c,cur_h) = self.decode_hidden_state
            self.c = tf.concat([self.c,cur_c],axis=3)
            self.h = tf.concat([self.h,cur_h],axis=3)
            self.final_input_state = tf.concat([self.final_input_state,tf.expand_dims(self.decode_input_state,axis=0)],axis=1)
            print 'output',self.final_input_state.get_shape()
            print 'c',self.c.get_shape()
            print 'h',self.h.get_shape()
        self.generated_images = self._decoder(tf.squeeze(self.final_input_state), trainable = self.trainable)
        self.original_images = self._decoder(tf.squeeze(self.input_state), trainable = self.trainable,scope_reuse = True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1 = 0.9)
        self.train_variables = tf.trainable_variables()

        self.reg1_loss = 10*tf.reduce_mean(tf.abs(self.input_images - self.original_images))
        self.reg2_loss = 10*tf.reduce_mean(tf.abs(self.input_images[1:,] - self.generated_images[:-1,]))+tf.reduce_mean(tf.abs(self.input_state[:,1:,] - self.final_input_state[:,:-1,]))
        self.reg1_variables = [v for v in self.train_variables if v.name.startswith("encoder") or v.name.startswith("decoder")]
        self.reg2_variables = [v for v in self.train_variables if v.name.startswith("ConvLSTMEncode") or v.name.startswith("ConvLSTMDecode") or v.name.startswith("encode_lstm_c") or v.name.startswith("encode_lstm_h") or v.name.startswith("encoder_detw")]
        print [v.name for v in self.train_variables]
        print [v.name for v in self.reg1_variables]
        print [v.name for v in self.reg2_variables]
        self.reg1_grads = self.optimizer.compute_gradients(self.reg1_loss, var_list=self.reg1_variables)
        self.reg1_op = self.optimizer.apply_gradients(self.reg1_grads)
        self.reg2_grads = self.optimizer.compute_gradients(self.reg2_loss, var_list=self.reg2_variables)
        self.reg2_op = self.optimizer.apply_gradients(self.reg2_grads)
        tf.summary.scalar("reg1_loss", self.reg1_loss)
        tf.summary.scalar("reg2_loss", self.reg2_loss)
    def _save_images(self,input_images,prefix):
        for i in range(self.batch_size):
            input_images[i] *= 127.5
            input_images[i] += 127.5
            input_images[i] = np.clip(input_images[i], 0, 255).astype(np.uint8)
            input_images[i] = np.reshape(input_images[i], (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, -1))
            misc.imsave(self.logs_dir+prefix+"_"+str(i)+".jpg", input_images[i])
        
    def _write_images(self,input_images,prefix):
        for i in range(self.batch_size/2):
            input_images[i+15] *= 127.5
            input_images[i+15] += 127.5
            input_images[i+15] = np.clip(input_images[i+15], 0, 255).astype(np.uint8)
            input_images[i+15] = np.reshape(input_images[i+15], (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, -1))
            misc.imsave("/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/DataSet/CityScape/CityScapeGen256128/"+str(i+prefix+15)+".png", input_images[i+15])
    def train(self):
        self.config = tf.ConfigProto()  
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.config)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        #params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #saver_params={}
        #reader = pywrap_tensorflow.NewCheckpointReader('/home/xjwxjw/Documents/DualSpaceTranformation/Test/Naive5_Pretrain_Feat/model.ckpt-6099')  
        #var_to_shape_map = reader.get_variable_to_shape_map()  
        #checkpoint_keys=var_to_shape_map.keys()
        #for v in params:
        #    v_name=v.name.split(':')[0]
        #    if v_name in checkpoint_keys:
        #        saver_params[v_name] = v
        #        print 'dec params: ',v_name
        #saver_res=tf.train.Saver(saver_params)
        #saver_res.restore(self.sess,'/home/xjwxjw/Documents/DualSpaceTranformation/Test/Naive5_Pretrain_Feat/model.ckpt-6099')
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(self.iterations):
            cur_dir = np.random.choice(600)
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.imglist[cur_dir*30+i]))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            _, reg1_loss_eval = self.sess.run([self.reg1_op,self.reg1_loss],feed_dict={self.input_images:img_batch})
            _, reg2_loss_eval = self.sess.run([self.reg2_op,self.reg2_loss],feed_dict={self.input_images:img_batch})
            print str(itr)+'reg1: '+str(reg1_loss_eval)+'reg2: '+str(reg2_loss_eval)
            if itr % 100 == 99:
                original_images_eval,gen_images_eval = self.sess.run([self.original_images,self.generated_images],feed_dict={self.input_images:img_batch})
                self._save_images(img_batch,"/train/int")
                self._save_images(original_images_eval,"/train/ori")
                self._save_images(gen_images_eval,"/train/gen")
                summary_str = self.sess.run(self.summary_op,feed_dict={self.input_images:img_batch})
                self.summary_writer.add_summary(summary_str, itr)
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
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(1):
            cur_dir = 374                          
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.imglist[cur_dir*30+self.batch_size-1 - i]))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            #reg1_loss_eval = self.sess.run(self.reg1_loss,feed_dict={self.input_images:img_batch})
            #reg2_loss_eval = self.sess.run(self.reg2_loss,feed_dict={self.input_images:img_batch})
            #print str(itr)+'reg1: '+str(reg1_loss_eval)+'reg2: '+str(reg2_loss_eval)
            gen_images_eval = self.sess.run(self.generated_images,feed_dict={self.input_images:img_batch})
            #if not os.path.exists(self.logs_dir+'/val/'+str(itr)):
            #    os.mkdir(self.logs_dir+'/val/'+str(itr))
            #    self._save_images(img_batch,"/val/"+str(itr)+"/input")
            #    self._save_images(original_images_eval,"/val/"+str(itr)+"/ori")
            self._save_images(gen_images_eval,"/val/gen")
            #    self._write_images(gen_images_eval,(itr+600)*30)
        coord.request_stop()
        coord.join(threads)
def main(argv=None):
    model = AutoEncoder(True)
    model.construct_network()
    #model.train()
    model.test()
if __name__ == "__main__":
    tf.app.run()

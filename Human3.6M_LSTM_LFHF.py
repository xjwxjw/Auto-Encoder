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
from cell_sigmoid_Human36M_LSTM_LFHF import ConvLSTMCell

def conv(batch_input, out_channels, kernel,stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel[0], kernel[1], in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [int((kernel[0]-1)/2), int((kernel[0]-1)/2)], [int((kernel[1]-1)/2), int((kernel[1]-1)/2)], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conved

def encoder_layer(input, leaky_rate = 0.2, out_channels = 64, kernel = [3,3], stride = 2, scope_name = 'encoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    convolved = conv(rectified, out_channels, kernel = kernel,stride=stride,scope_name = scope_name)
    #output = batchnorm(convolved, scope_name = scope_name)
    #output = tf.contrib.layers.batch_norm(convolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return convolved

def deconv(batch_input, out_channels,kernel,stride,scope_name):
    with tf.variable_scope("deconv-" + scope_name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel[0], kernel[1], out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        deconved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        return deconved

def decoder_layer(input, out_channels = 64, kernel = [2,2],stride = 2, scope_name = 'decoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    deconvolved = deconv(rectified, out_channels, kernel = kernel,stride = stride,scope_name = scope_name)
    return deconvolved

start_order = [1,21,12,36,1,12,7,1,17,1,1,27,27,47,16,9,8,6,1]
dir_name = ['1_1','1_2','5_1','5_2','6_1','6_2','7_1','7_2','8_1','8_2','9_1','9_2','11_1','11_2']
class AutoEncoder(object):
    def __init__(self, trainable):
        self.batch_size = 30
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.IMAGE_CHANNEL = 3
        self.iterations = 500000
        self.lamda_dis = 0
        self.lamda_reg = 100
        self.learning_rate = 1e-4
        self.logs_dir = "/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/Experiment/Result/Human3.6M_LSTM_LFHF_Consecutive"
        self.hidden_state_dir = "/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/DataSet/SnatchReal_Hidden_State_128/"
        self.images_dir = "/home/xjwxjw/deva/Human3.6M_Cordinate/Human_img_data/Walking_S"
        self.trainable = trainable
        self.rnn_unit = 1024
        self.time_step = 30
        self.conLSTM_kernel = [5,5]
        self.dir_list = []
        self.dir_start = []
        self.lstm_channel = 32
        if not os.path.exists(os.path.join(self.logs_dir)):
            os.mkdir(os.path.join(self.logs_dir))
        if not os.path.exists(os.path.join(self.logs_dir,'train')):
            os.mkdir(os.path.join(self.logs_dir,'train'))
        if not os.path.exists(os.path.join(self.logs_dir,'val')):
            os.mkdir(os.path.join(self.logs_dir,'val'))
    def _convLSTM(self,input_hidden_state, scope_name = 'convLSTM', initial_state=None,trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            # Create a placeholder for videos.
            print scope_name,input_hidden_state.get_shape()
            cell = ConvLSTMCell([self.IMAGE_HEIGHT/4,self.IMAGE_WIDTH/4], self.lstm_channel, self.conLSTM_kernel)
            if initial_state == None:
                 outputs, state = tf.nn.dynamic_rnn(cell, input_hidden_state, initial_state=cell.zero_state(1,dtype=tf.float32),dtype=input_hidden_state.dtype)
            else:
                outputs, state = tf.nn.dynamic_rnn(cell, input_hidden_state, initial_state=initial_state,dtype=input_hidden_state.dtype)
            print scope_name,outputs.get_shape()
            return outputs, state
    def _encode_lstm_hidden(self,input_hidden_state, scope_name = 'encode_lstm', initial_state=None,trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            print 'encode_lstm_hidden_in',input_hidden_state.get_shape()
            output = encoder_layer(input_hidden_state, leaky_rate = 0.2, out_channels = self.lstm_channel, stride = 1, scope_name = 'encoder_lstm', trainable = trainable)#(32, 32, 512)==>(32, 32, 128)
            #output = tf.expand_dims(output,axis=0)
            print 'encode_lstm_hidden_out',output.get_shape()   
            return output
    def _encoder(self, input_images, scope_name = "encoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            output1 = conv(input_images, 8, stride= 2, kernel = [5,5],scope_name = 'encoder_layer1')#(128, 256, 19)==>(64, 128, 19)
            print output1.get_shape()
            tf.summary.histogram(output1.op.name + "/activation", output1)

            output2 = encoder_layer(output1, leaky_rate = 0.2, out_channels = 16, stride = 2, kernel = [3,3],scope_name = 'encoder_layer2', trainable = trainable)#(64, 128, 19)==>(32, 64, 19)
            print output2.get_shape()   
            tf.summary.histogram(output2.op.name + "/activation", output2)
         
            output3 = encoder_layer(output2, leaky_rate = 0.2, out_channels = 32, stride = 1, kernel = [3,3],scope_name = 'encoder_layer3', trainable = trainable)#(32, 64, 19)==>(16, 32, 19)
            print output3.get_shape()
            tf.summary.histogram(output3.op.name + "/activation", output3)
        return tf.nn.softplus(output3)
    def _decoder(self, input_hidden_state, scope_name = "decoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            output4 = decoder_layer(input_hidden_state, out_channels = 16, stride = 1, kernel = [3,3],scope_name = 'decoder_layer4', trainable = trainable)#(16, 32, 19)==>(32, 64, 19)
            print output4.get_shape()
            tf.summary.histogram(output4.op.name + "/activation", output4)

            output5 = decoder_layer(output4, out_channels = 8, stride = 2, kernel = [3,3],scope_name = 'decoder_layer5', trainable = trainable)#(32, 64, 19)==>(64, 128, 19)
            print output5.get_shape()
            tf.summary.histogram(output5.op.name + "/activation", output5)
            
            generated_images = decoder_layer(output5, out_channels = 3, stride = 2, kernel = [5,5],scope_name = 'decoder_layer6', trainable = trainable)#(64, 128, 19)==>(128, 256, 19)
            print generated_images.get_shape()
            tf.summary.histogram(generated_images.op.name + "/activation", generated_images)
            return generated_images
    def SeqToSeqLSTM(self,input_hidden_state,scope_name = "SeqToSeqLSTM", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            input_state = tf.expand_dims(input_hidden_state,axis=0)
            encode_input_state,encode_hidden_state = self._convLSTM(tf.expand_dims(tf.expand_dims(input_state[0,0],axis=0),axis=0),scope_name = 'ConvLSTMEncode')
            final_input_state = encode_input_state
            (c,h) = encode_hidden_state
            print 'output',final_input_state.get_shape()
            print 'c',c.get_shape()
            print 'h',h.get_shape()
            for i in range(self.time_step/2-1):
                encode_input_state,encode_hidden_state = self._convLSTM(tf.expand_dims(tf.expand_dims(input_state[0,i+1],axis=0),axis=0),initial_state=encode_hidden_state,scope_name = 'ConvLSTMEncode',scope_reuse=True)
                (cur_c,cur_h) = encode_hidden_state
                c = tf.concat([c,cur_c],axis=3)
                h = tf.concat([h,cur_h],axis=3)
                final_input_state = tf.concat([final_input_state,encode_input_state],axis=1)
                print 'output',final_input_state.get_shape()
                print 'c',c.get_shape()
                print 'h',h.get_shape()

            print 'last4c',c[0,:,:,-4*self.lstm_channel:].get_shape()
            print 'last4h',h[0,:,:,-4*self.lstm_channel:].get_shape()
            last4_c = self._encode_lstm_hidden(tf.expand_dims(c[0,:,:,-4*self.lstm_channel:],axis=0), scope_name = 'encode_lstm_c')
            last4_h = self._encode_lstm_hidden(tf.expand_dims(h[0,:,:,-4*self.lstm_channel:],axis=0), scope_name = 'encode_lstm_h')
            last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(last4_c,last4_h)
            decode_input_state = encode_input_state
            decode_input_state,decode_hidden_state = self._convLSTM(decode_input_state,initial_state=last4_hidden_state,scope_name = 'ConvLSTMDecode')
            #decode_input_state,decode_hidden_state = self._convLSTM(tf.expand_dims(tf.expand_dims(input_state[0,15],axis=0),axis=0),initial_state=last4_hidden_state,scope_name = 'ConvLSTMDecode') 
            (cur_c,cur_h) = decode_hidden_state
            c = tf.concat([c,cur_c],axis=3)
            h = tf.concat([h,cur_h],axis=3)
            final_input_state = tf.concat([final_input_state,decode_input_state],axis=1)
            print 'output',final_input_state.get_shape()
            print 'c',c.get_shape()
            print 'h',h.get_shape()
            print final_input_state.get_shape()
            for i in range(self.time_step/2-1):
                last4_c = self._encode_lstm_hidden(tf.expand_dims(c[0,:,:,-4*self.lstm_channel:],axis=0), scope_name = 'encode_lstm_c',scope_reuse=True)
                last4_h = self._encode_lstm_hidden(tf.expand_dims(h[0,:,:,-4*self.lstm_channel:],axis=0), scope_name = 'encode_lstm_h',scope_reuse=True)
                last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(last4_c,last4_h)
                decode_input_state,decode_hidden_state = self._convLSTM(decode_input_state,initial_state=last4_hidden_state,scope_name = 'ConvLSTMDecode',scope_reuse=True)
                #decode_input_state,decode_hidden_state = self._convLSTM(tf.expand_dims(tf.expand_dims(input_state[0,16+i],axis=0),axis=0),initial_state=last4_hidden_state,scope_name = 'ConvLSTMDecode',scope_reuse=True)
                (cur_c,cur_h) = decode_hidden_state
                c = tf.concat([c,cur_c],axis=3)
                h = tf.concat([h,cur_h],axis=3)
                final_input_state = tf.concat([final_input_state,decode_input_state],axis=1)
                print 'output',final_input_state.get_shape()
                print 'c',c.get_shape()
                print 'h',h.get_shape()
            return final_input_state
    def construct_network(self):
        self.input_images = tf.placeholder(tf.float32, [self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.IMAGE_CHANNEL], name="input_images")
        self.padded_input_images = tf.pad(self.input_images, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
        self.LoG = tf.cast(tf.constant(sio.loadmat('../Utils/LoG3x3.mat')['LoG']),tf.float32)
        self.hf_input_images = tf.nn.conv2d(self.padded_input_images, self.LoG, [1, 1, 1, 1], padding="VALID")/32.0
        self.input_state = self._encoder(self.input_images,scope_name = "lfencoder",trainable = self.trainable) 
        self.hf_input_state = self._encoder(self.hf_input_images,scope_name = "hfencoder",trainable = self.trainable)
        self.original_images = self._decoder(tf.concat([self.input_state,self.hf_input_state],-1), trainable = self.trainable)
        print "LL",self.hf_input_state.get_shape()
        print "KK",self.input_state.get_shape()
        self.final_input_state = self.SeqToSeqLSTM(self.input_state,scope_name = "LFSeqToSeqLSTM")
        self.hf_final_input_state = self.SeqToSeqLSTM(self.hf_input_state,scope_name = "HFSeqToSeqLSTM")
        print self.final_input_state.get_shape()
        print self.hf_final_input_state.get_shape()
        self.final_input_state = tf.squeeze(self.final_input_state,0)
        self.hf_final_input_state = tf.squeeze(self.hf_final_input_state,0)
        self.generated_images = self._decoder(tf.concat([self.final_input_state,self.hf_final_input_state],-1), scope_reuse = True)
        self.train_variables = tf.trainable_variables()

        ########################High Freuqncy Loss#####################################
        padded_ori = tf.pad(self.original_images, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
        self.hf_ori = tf.nn.conv2d(padded_ori, self.LoG, [1, 1, 1, 1], padding="VALID")/32.0
        self.recon_ori_loss = tf.reduce_mean(tf.abs(self.input_images-self.original_images))#==>lfencoder+hfencoder+decoder
        self.hf_ori_loss = tf.reduce_mean(tf.abs(self.hf_ori-self.hf_input_images))#==>lfencoder+hfencoder+decoder

        padded_gen = tf.pad(self.generated_images, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
        self.hf_gen = tf.nn.conv2d(padded_gen, self.LoG, [1, 1, 1, 1], padding="VALID")/32.0
        self.recon_gen_loss = tf.reduce_mean(tf.abs(self.input_images[1:] - self.generated_images[:-1]))
        self.hf_gen_loss = tf.reduce_mean(tf.abs(self.hf_input_images[1:] - self.hf_gen[:-1]))
        self.state_loss = tf.reduce_mean(tf.abs(self.input_state[:,1:,] - self.final_input_state[:,:-1,]))
        self.hf_state_loss = tf.reduce_mean(tf.abs(self.hf_input_state[:,1:,] - self.hf_final_input_state[:,:-1,]))
        ########################High Freuqncy Loss#####################################

        self.optimizer = tf.train.AdamOptimizer(1e-4,beta1 = 0.9)
        self.train_variables = tf.trainable_variables()
        self.AutoEncoder_variables = [v for v in self.train_variables if v.name.startswith("lfencoder") or v.name.startswith("decoder") or v.name.startswith("hfencoder")]
        self.LSTM_variables = [v for v in self.train_variables if v.name.startswith("LFSeqToSeqLSTM") or v.name.startswith("HFSeqToSeqLSTM")]
        self.LFLSTM_variables = [v for v in self.train_variables if v.name.startswith("LFSeqToSeqLSTM")]
        self.HFLSTM_variables = [v for v in self.train_variables if v.name.startswith("HFSeqToSeqLSTM")]

        self.recon_ori_grads = self.optimizer.compute_gradients(self.recon_ori_loss, var_list=self.AutoEncoder_variables)
        self.recon_ori_op = self.optimizer.apply_gradients(self.recon_ori_grads)
        self.hf_ori_grads = self.optimizer.compute_gradients(self.hf_ori_loss, var_list=self.AutoEncoder_variables)
        self.hf_ori_op = self.optimizer.apply_gradients(self.hf_ori_grads)
        self.recon_gen_grads = self.optimizer.compute_gradients(self.recon_gen_loss, var_list=self.LSTM_variables)
        self.recon_gen_op = self.optimizer.apply_gradients(self.recon_ori_grads)
        self.hf_gen_grads = self.optimizer.compute_gradients(self.hf_gen_loss, var_list=self.LSTM_variables)
        self.hf_gen_op = self.optimizer.apply_gradients(self.hf_gen_grads)
        self.state_grads = self.optimizer.compute_gradients(self.state_loss, var_list=self.LFLSTM_variables)
        self.state_op = self.optimizer.apply_gradients(self.state_grads)
        self.hf_state_grads = self.optimizer.compute_gradients(self.hf_state_loss, var_list=self.HFLSTM_variables)
        self.hf_state_op = self.optimizer.apply_gradients(self.hf_state_grads)
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
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver_params={}
        reader = pywrap_tensorflow.NewCheckpointReader('/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/Experiment/Result/Human3.6M_LFHF/model.ckpt-28099')  
        var_to_shape_map = reader.get_variable_to_shape_map()  
        checkpoint_keys=var_to_shape_map.keys()
        for v in params:
            v_name=v.name.split(':')[0]
            if v_name in checkpoint_keys:
                saver_params[v_name] = v
                print 'dec params: ',v_name
        saver_res=tf.train.Saver(saver_params)
        saver_res.restore(self.sess,'/media/xjwxjw/TOSHIBA EXT/DualSapceTransformation/Experiment/Result/Human3.6M_LFHF/model.ckpt-28099')
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(self.iterations):
            cur_dir1 = np.random.choice(14)
            cur_dir2 = np.random.choice(1471)+1
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.images_dir+dir_name[cur_dir1]+'/'+str(cur_dir2+20 - i)+'.jpg'))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            _, recon_ori_loss_eval, _, hf_ori_loss_eval = self.sess.run([self.recon_ori_op,self.recon_ori_loss,self.hf_ori_op,self.hf_ori_loss],feed_dict={self.input_images:img_batch})
            _, recon_gen_loss_eval, _, hf_gen_loss_eval = self.sess.run([self.recon_gen_op,self.recon_gen_loss,self.hf_gen_op,self.hf_gen_loss],feed_dict={self.input_images:img_batch})
            _, state_loss_eval, _, hf_state_loss_eval = self.sess.run([self.state_op,self.state_loss,self.hf_state_op,self.hf_state_loss],feed_dict={self.input_images:img_batch})
            print "Snatch:"+str(itr)+'recon_ori: '+str(recon_ori_loss_eval)+'hf_ori: '+str(hf_ori_loss_eval)+'recon_gen: '+str(recon_gen_loss_eval)+'hf_gen: '+str(hf_gen_loss_eval)\
                          +'state: '+str(state_loss_eval)+'hf_state: '+str(hf_state_loss_eval)
            if itr % 100 == 99:
                original_images_eval,gen_images_eval = self.sess.run([self.original_images,self.generated_images],feed_dict={self.input_images:img_batch})
                if not os.path.exists(self.logs_dir+'/train/'+str(itr)):
                    os.mkdir(self.logs_dir+'/train/'+str(itr))
                self._save_images(img_batch,"/train/"+str(itr)+"/input_")
                self._save_images(original_images_eval,"/train/"+str(itr)+"/ori1_")
                self._save_images(gen_images_eval,"/train/"+str(itr)+"/gen1_")
                cur_dir1 = np.random.choice(14)
                cur_dir2 = np.random.choice(1471)+1
                for i in range(self.batch_size):
                    cur_img = np.float32(misc.imread(self.images_dir+dir_name[cur_dir1]+'/'+str(cur_dir2+i)+'.jpg'))
                    cur_img -= 127.5
                    cur_img /= 127.5
                    img_batch[i] = cur_img
                original_images_eval,gen_images_eval = self.sess.run([self.original_images,self.generated_images],feed_dict={self.input_images:img_batch})
                if not os.path.exists(self.logs_dir+'/val/'+str(itr)):
                    os.mkdir(self.logs_dir+'/val/'+str(itr))
                self._save_images(img_batch,"/val/"+str(itr)+"/input_")
                self._save_images(original_images_eval,"/val/"+str(itr)+"/ori1_")
                self._save_images(gen_images_eval,"/val/"+str(itr)+"/gen1_")
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
        for itr in range(60):
            print itr
            cur_dir1 = 6#np.random.choice(14)
            cur_dir2 = 354+itr#np.random.choice(1471)+1
            for i in range(self.batch_size):
                cur_img = np.float32(misc.imread(self.images_dir+dir_name[cur_dir1]+'/'+str(cur_dir2+self.batch_size - i)+'.jpg'))
                cur_img -= 127.5
                cur_img /= 127.5
                img_batch[i] = cur_img
            original_images_eval,gen_images_eval = self.sess.run([self.original_images,self.generated_images],feed_dict={self.input_images:img_batch})
            if not os.path.exists(self.logs_dir+'/val/'+str(itr)+'past'):
                os.mkdir(self.logs_dir+'/val/'+str(itr)+'past')
            self._save_images(img_batch,"/val/"+str(itr)+"past"+"/input_")
            self._save_images(original_images_eval,"/val/"+str(itr)+"past"+"/ori1_")
            self._save_images(gen_images_eval,"/val/"+str(itr)+"past"+"/gen1_")
            #summary_str = self.sess.run(self.summary_op,feed_dict={self.input_images:img_batch})
            #self.summary_writer.add_summary(summary_str, itr)
        coord.request_stop()
        coord.join(threads)
def main(argv=None):
    model = AutoEncoder(True)
    model.construct_network()
    #model.train()
    model.test()
if __name__ == "__main__":
    tf.app.run()

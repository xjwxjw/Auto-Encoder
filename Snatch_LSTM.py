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
from cell_Snatch_LSTM import ConvLSTMCell

def conv(batch_input, out_channels, stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
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

start_order = [21,51,51,39,59,19,49,39,19,49,29,29,49,29,39,39,49,29,29]
dir_name = ['03_01','03_02','03_03','03_04','03_05','03_06','07_01','07_02','07_03','07_04','07_05','11_01','11_02','11_03','11_04','18_01','18_02','18_03','18_04']
class AutoEncoder(object):
    def __init__(self, trainable):
        self.batch_size = 32
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.NUM_CHANNELS = 3
        self.iterations = 500000
        self.lamda_dis = 0
        self.lamda_reg = 100
        self.learning_rate = 1e-4
        self.logs_dir = "./Snatch_ConvLSTM_down_"+str(self.lamda_reg)+'_'+str(self.lamda_dis)+'_'+str(self.learning_rate)
        self.hidden_state_dir = "/home/xjwxjw/Documents/DualSpaceTranformation/SnatchDataset/SegImg/Val/21/"
        self.images_dir = "/home/xjwxjw/Documents/DualSpaceTranformation/SnatchDataset/Naive4/"
        self.trainable = trainable
        self.rnn_unit = 1024
        self.time_step = 32
        self.conLSTM_kernel = [5,5]
        self.dir_list = []
        self.dir_start = []
        self.lstm_channel = 128
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
            output = encoder_layer(input_hidden_state, leaky_rate = 0.2, out_channels = 128, stride = 1, scope_name = 'encoder_lstm', trainable = trainable)#(32, 32, 512)==>(32, 32, 128)
            #output = tf.expand_dims(output,axis=0)
            print 'encode_lstm_hidden_out',output.get_shape()   
            return output
    def _encoder(self, input_images, scope_name = "encoder", trainable = True, scope_reuse = False):
        with arg_scope(resnet_utils.resnet_arg_scope()):
            output, end_points = resnet_v2.resnet_v2_50(input_images, output_stride=8, global_pool=False,reuse=scope_reuse)#(256, 256, 2048)==>(32, 32, 2048)
            hidden_state = decoder_layer(output, out_channels = self.lstm_channel, stride = 1, scope_name = 'encoder_layer1', trainable = trainable)#(32, 32, 2048)==>(32, 32, 512)
            print hidden_state.get_shape()
            tf.summary.histogram(hidden_state.op.name + "/activation", hidden_state)
            return hidden_state
    def _decoder(self, input_hidden_state, scope_name = "decoder", trainable = True, scope_reuse = False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            #input_hidden_state = tf.sigmoid(input_hidden_state)
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
        self.input_state = tf.placeholder(tf.float32, [1, self.time_step] + [self.IMAGE_HEIGHT/8,self.IMAGE_WIDTH/8] + [self.lstm_channel])
        self.encode_input_state,self.encode_hidden_state,_ = self._convLSTM(tf.expand_dims(tf.expand_dims(self.input_state[0,0],axis=0),axis=0),scope_name = 'ConvLSTMEncode')
        self.final_input_state = self.encode_input_state
        (self.c,self.h) = self.encode_hidden_state
        print 'output',self.final_input_state.get_shape()
        print 'c',self.c.get_shape()
        print 'h',self.h.get_shape()
        for i in range(15):
            self.encode_input_state,self.encode_hidden_state,_ = self._convLSTM(tf.expand_dims(tf.expand_dims(self.input_state[0,i+1],axis=0),axis=0),initial_state=self.encode_hidden_state,scope_name = 'ConvLSTMEncode',scope_reuse=True)
            (cur_c,cur_h) = self.encode_hidden_state
            self.c = tf.concat([self.c,cur_c],axis=3)
            self.h = tf.concat([self.h,cur_h],axis=3)
            self.final_input_state = tf.concat([self.final_input_state,self.encode_input_state],axis=1)
            print 'output',self.final_input_state.get_shape()
            print 'c',self.c.get_shape()
            print 'h',self.h.get_shape()

        print 'last4c',self.c[0,:,:,-512:].get_shape()
        print 'last4h',self.h[0,:,:,-512:].get_shape()
        self.last4_c = self._encode_lstm_hidden(tf.expand_dims(self.c[0,:,:,-512:],axis=0), scope_name = 'encode_lstm_c')
        self.last4_h = self._encode_lstm_hidden(tf.expand_dims(self.h[0,:,:,-512:],axis=0), scope_name = 'encode_lstm_h')
        self.last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(self.last4_c,self.last4_h)
        self.decode_input_state = self.encode_input_state
        self.decode_input_state,self.decode_hidden_state,_ = self._convLSTM(self.decode_input_state,initial_state=self.last4_hidden_state,scope_name = 'ConvLSTMDecode')
        (cur_c,cur_h) = self.decode_hidden_state
        self.c = tf.concat([self.c,cur_c],axis=3)
        self.h = tf.concat([self.h,cur_h],axis=3)
        self.final_input_state = tf.concat([self.final_input_state,self.decode_input_state],axis=1)
        print 'output',self.final_input_state.get_shape()
        print 'c',self.c.get_shape()
        print 'h',self.h.get_shape()
        print self.final_input_state.get_shape()
        for i in range(15):
            self.last4_c = self._encode_lstm_hidden(tf.expand_dims(self.c[0,:,:,-512:],axis=0), scope_name = 'encode_lstm_c',scope_reuse=True)
            self.last4_h = self._encode_lstm_hidden(tf.expand_dims(self.h[0,:,:,-512:],axis=0), scope_name = 'encode_lstm_h',scope_reuse=True)
            self.last4_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(self.last4_c,self.last4_h)
            self.decode_input_state,self.decode_hidden_state,_ = self._convLSTM(self.decode_input_state,initial_state=self.last4_hidden_state,scope_name = 'ConvLSTMDecode',scope_reuse=True)
            (cur_c,cur_h) = self.decode_hidden_state
            self.c = tf.concat([self.c,cur_c],axis=3)
            self.h = tf.concat([self.h,cur_h],axis=3)
            self.final_input_state = tf.concat([self.final_input_state,self.decode_input_state],axis=1)
            print 'output',self.final_input_state.get_shape()
            print 'c',self.c.get_shape()
            print 'h',self.h.get_shape()
        _,_,self.lstm_W = self._convLSTM(self.decode_input_state,initial_state=self.last4_hidden_state,scope_name = 'ConvLSTMDecode',scope_reuse=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1 = 0.9)
        self.train_variables = tf.trainable_variables()
        #self.fake_features = self._discriminator(tf.squeeze(self.final_input_state))
        #self.real_features = self._discriminator(tf.squeeze(self.input_state),scope_reuse = True)
        #self.dis_loss = self.lamda_dis * tf.reduce_mean(self.fake_features[:-1,] - self.real_features[1:,])
        #self.dis_variables = [v for v in self.train_variables if v.name.startswith("ConvLSTMEncode") or v.name.startswith("ConvLSTMDecode") or v.name.startswith("encode_lstm_c") or v.name.startswith("encode_lstm_h") or v.name.startswith("discriminator")]
        #print self.dis_variables
        #self.dis_grads = self.optimizer.compute_gradients(self.dis_loss,var_list = self.dis_variables)
        #self.dis_op = self.optimizer.apply_gradients(self.dis_grads)
        self.generated_images = self._decoder(tf.squeeze(self.final_input_state), trainable = self.trainable)
        self.original_images = self._decoder(tf.squeeze(self.input_state), trainable = self.trainable,scope_reuse = True)
        #self.reg_loss = self.lamda_reg * tf.reduce_mean(tf.matmul(tf.squeeze(self.input_state[:,1:,]),tf.squeeze(self.input_state[:,1:,]),transpose_b=True) - \
        #                                                tf.matmul(tf.squeeze(self.final_input_state[:,:-1,]),tf.squeeze(self.final_input_state[:,:-1,]),transpose_b=True))
        self.reg_loss = self.lamda_reg * (tf.reduce_mean(tf.abs(self.input_state[:,1:,] - self.final_input_state[:,:-1,])))
        self.reg_variables = [v for v in self.train_variables if v.name.startswith("ConvLSTMEncode") or v.name.startswith("ConvLSTMDecode") or v.name.startswith("encode_lstm_c") or v.name.startswith("encode_lstm_h")]
        print self.reg_variables
        
        self.reg_grads = self.optimizer.compute_gradients(self.reg_loss, var_list=self.reg_variables)
        self.reg_op = self.optimizer.apply_gradients(self.reg_grads)
        tf.summary.scalar("reg_loss", self.reg_loss)
        #tf.summary.scalar("dis_loss", self.dis_loss)
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
        reader = pywrap_tensorflow.NewCheckpointReader('/home/xjwxjw/Documents/DualSpaceTranformation/Test/Snatch_1_32_32_Feat/model.ckpt-14099')  
        var_to_shape_map = reader.get_variable_to_shape_map()  
        checkpoint_keys=var_to_shape_map.keys()
        for v in params:
            v_name=v.name.split(':')[0]
            if v_name in checkpoint_keys:
                saver_params[v_name] = v
                print 'dec params: ',v_name
        saver_res=tf.train.Saver(saver_params)
        saver_res.restore(self.sess,'/home/xjwxjw/Documents/DualSpaceTranformation/Test/Snatch_1_32_32_Feat/model.ckpt-14099')
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Model Restoring..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(self.iterations):
            cur_dir1 = np.random.choice(19)
            #while((cur_dir1+1)%19 == 0):
            #    cur_dir1 = np.random.choice(1900)
            hidden_state_batch = sio.loadmat(self.hidden_state_dir+'/feature_down_'+str(cur_dir1)+'.mat')
            #hidden_state_batch = sio.loadmat('/home/xjwxjw/Documents/DualSpaceTranformation/Test/Naive4_AutoRes_ConvLSTM_0_1_0.0001/feature_train.mat')
            _, reg_loss_eval = self.sess.run([self.reg_op,self.reg_loss],feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
            #_, dis_loss_eval = self.sess.run([self.dis_op,self.dis_loss],feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
            print str(itr)+'reg: '+str(reg_loss_eval)#+'dis: '+str(dis_loss_eval)
            if itr % 100 == 99:
                original_images_eval,gen_images_eval,final_input_state_eval,c_eval,h_eval = self.sess.run([self.original_images,self.generated_images,\
                       tf.squeeze(self.final_input_state),tf.squeeze(self.c),tf.squeeze(self.h)],feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
                sio.savemat(self.logs_dir+'/feature_train.mat', {'feature_train': final_input_state_eval})
                sio.savemat(self.logs_dir+'/c_train.mat', {'c_train': c_eval})
                sio.savemat(self.logs_dir+'/h_train.mat', {'h_train': h_eval})
                if not os.path.exists(self.logs_dir+'/train/'+str(itr)):
                    os.mkdir(self.logs_dir+'/train/'+str(itr))
                self._save_images(original_images_eval,"/train/"+str(itr)+"/ori_")
                self._save_images(gen_images_eval,"/train/"+str(itr)+"/gen_")
                #cur_dir1 = np.random.choice(100)*19 + 18
                #hidden_state_batch = sio.loadmat(self.hidden_state_dir+'/feature_'+str(cur_dir1)+'.mat')
                #original_images_eval,gen_images_eval,final_input_state_eval,c_eval,h_eval = self.sess.run([self.original_images,self.generated_images,\
                #       tf.squeeze(self.final_input_state),tf.squeeze(self.c),tf.squeeze(self.h)],feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
                #sio.savemat(self.logs_dir+'/feature_val.mat', {'feature_val': final_input_state_eval})
                #sio.savemat(self.logs_dir+'/c_val.mat', {'c_val': c_eval})
                #sio.savemat(self.logs_dir+'/h_val.mat', {'h_val': h_eval})
                #self._save_images(original_images_eval,"/val/ori1_"+str(0))
                #self._save_images(gen_images_eval,"/val/gen1_"+str(0))
                summary_str = self.sess.run(self.summary_op,feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
                self.summary_writer.add_summary(summary_str, itr)
                #encode_hidden_state_eval = self.sess.run([tf.squeeze(self.encode_hidden_state)],\
                #                         feed_dict={self.hidden_state:np.expand_dims(hidden_state_batch['feature_train'],axis=0)})
                #sio.savemat(self.logs_dir+'/feature_val.mat', {'feature_val': encode_hidden_state_eval})
            if itr % 100 == 99:
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

        #params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #saver_params={}
        #reader = pywrap_tensorflow.NewCheckpointReader('/home/xjwxjw/Documents/DualSpaceTranformation/Test/Snatch_1_32_32_Feat/model.ckpt-14099')  
        #var_to_shape_map = reader.get_variable_to_shape_map()  
        #checkpoint_keys=var_to_shape_map.keys()
        #for v in params:
        #    v_name=v.name.split(':')[0]
        #    if v_name in checkpoint_keys:
        #        saver_params[v_name] = v
        #        print 'dec params: ',v_name
        #saver_res=tf.train.Saver(saver_params)
        #saver_res.restore(self.sess,'/home/xjwxjw/Documents/DualSpaceTranformation/Test/Snatch_1_32_32_Feat/model.ckpt-14099')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        print "LL"
        img_batch = np.zeros((self.batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH,3))
        for itr in range(1):
            print itr
            cur_dir1 = 10#np.random.choice(100)*19 + itr
            hidden_state_batch = sio.loadmat(self.hidden_state_dir+'/feature_down_'+str(cur_dir1)+'.mat')
            original_images_eval,gen_images_eval,final_input_state_eval,c_eval,h_eval = self.sess.run([self.original_images,self.generated_images,\
                       tf.squeeze(self.final_input_state),tf.squeeze(self.c),tf.squeeze(self.h)],feed_dict={self.input_state:np.expand_dims(hidden_state_batch['feature_val'],axis=0)})
            sio.savemat(self.logs_dir+'/feature_train.mat', {'feature_train': final_input_state_eval})
            sio.savemat(self.logs_dir+'/c_train.mat', {'c_train': c_eval})
            sio.savemat(self.logs_dir+'/h_train.mat', {'h_train': h_eval})
            if not os.path.exists(self.logs_dir+'/val/'+str(itr)):
                os.mkdir(self.logs_dir+'/val/'+str(itr))
            self._save_images(original_images_eval,"/val/"+str(itr)+"/ori_")
            self._save_images(gen_images_eval,"/val/"+str(itr)+"/gen_")
        coord.request_stop()
        coord.join(threads)
def main(argv=None):
    model = AutoEncoder(True)
    model.construct_network()
    #model.train()
    model.test()
if __name__ == "__main__":
    tf.app.run()

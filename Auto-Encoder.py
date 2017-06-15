import tensorflow as tf
import numpy as np
import scipy.misc as misc
from tensorflow.contrib import rnn
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
    output = batchnorm(convolved, scope_name = scope_name)
    #output = tf.contrib.layers.batch_norm(convolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return output

def decoder_layer(input, out_channels = 64, stride = 2, scope_name = 'decoder_layer', trainable = True):
    rectified = tf.nn.relu(input)
    deconvolved = deconv(rectified, out_channels, scope_name = scope_name)
    output = batchnorm(deconvolved, scope_name = scope_name, trainable = trainable)
    #output = tf.contrib.layers.batch_norm(deconvolved,decay = 0.9,center = True, scale = True, epsilon = 1e-5, is_training = True, trainable = trainable, scope = scope_name)
    return output

class AutoEncoder(object):
    def __init__(self, trainable):
        self.batch_size = 64
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.NUM_CHANNELS = 3
        self.iterations = 50000
        self.logs_dir = "./H3.6M/"
        self.learning_rate = 1e-4
        self.trainable = trainable
        self.rnn_unit = 1024
        self.time_step = 32
    def read_image(self):
        filename = []
        #for line in open('list.txt','r'):
        for i in range(1,1500):
            filename.append('./H3.6Mdata/'+str(i)+'.jpg')
            #filename.append(line.split('\n')[0])
        images_queue = tf.train.string_input_producer(filename,shuffle=False)
        reader = tf.WholeFileReader()
        key, value = reader.read(images_queue)
        decoded_image = tf.image.decode_jpeg(value,channels=3)
        decoded_image_4d = tf.expand_dims(decoded_image, 0)
        decoded_image = tf.image.resize_bilinear(decoded_image_4d, [self.IMAGE_WIDTH, self.IMAGE_WIDTH])
        squeezed_image = tf.squeeze(decoded_image, squeeze_dims=[0])
        squeezed_image.set_shape([self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_CHANNELS])
        float_image = (tf.cast(squeezed_image,tf.float32) - 127.5)/127.5
        input_image = tf.train.batch([float_image],batch_size = self.batch_size)
        return input_image
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
    def _encoder(self, input_images, scope_name = "encoder", trainable = True):
        hidden_state = []
        with tf.variable_scope(scope_name) as scope:
            print input_images.get_shape()
            output = conv(input_images, 8, stride=2 ,scope_name = 'encoder_layer1')#(256, 256, 3)==>(128, 128, 8)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 16, stride = 2, scope_name = 'encoder_layer2', trainable = trainable)#(128, 128, 8)==>(64, 64, 16)
            print output.get_shape()   
            tf.summary.histogram(output.op.name + "/activation", output)
         
            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 32, stride = 2, scope_name = 'encoder_layer3', trainable = trainable)#(64, 64, 16)==>(32, 32, 32)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 64, stride = 2, scope_name = 'encoder_layer4', trainable = trainable)#(32, 32, 32)==>(16, 16, 64)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 128, stride = 2, scope_name = 'encoder_layer5', trainable = trainable)#(16, 16, 64)==>(8, 8, 128)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 256, stride = 2, scope_name = 'encoder_layer6', trainable = trainable)#(8, 8, 128)==>(4, 4, 256)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)
 
            output = encoder_layer(output, leaky_rate = 0.2, out_channels = 512, stride = 2, scope_name = 'encoder_layer7', trainable = trainable)#(4, 4, 256)==>(2, 2, 512)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)
 
            hidden_state = encoder_layer(output, leaky_rate = 0.2, out_channels = 512, stride = 2, scope_name = 'encoder_layer8', trainable = trainable)#(2, 2, 512)==>(1, 1, 512)
            print hidden_state.get_shape()
            tf.summary.histogram(hidden_state.op.name + "/activation", hidden_state)
        return hidden_state
    def _decoder(self, input_hidden_state, scope_name = "decoder", trainable = True):
        with tf.variable_scope(scope_name) as scope:
            output = decoder_layer(input_hidden_state, out_channels = 256, stride = 2, scope_name = 'decoder_layer1', trainable = trainable)#(1, 1, 512)==>(2, 2, 256)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 256, stride = 2, scope_name = 'decoder_layer2', trainable = trainable)#(2, 2, 256)==>(4, 4, 256)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 256, stride = 2, scope_name = 'decoder_layer3', trainable = trainable)#(4, 4, 256)==>(8, 8, 256)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 128, stride = 2, scope_name = 'decoder_layer4', trainable = trainable)#(8, 8, 256)==>(16, 16, 128)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 64, stride = 2, scope_name = 'decoder_layer5', trainable = trainable)#(16, 16, 128)==>(32, 32, 64)
            print output.get_shape()
            tf.summary.histogram(output.op.name + "/activation", output)

            output = decoder_layer(output, out_channels = 32, stride = 2, scope_name = 'decoder_layer6', trainable = trainable)#(32, 32, 64)==>(64, 64, 32)
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
        self.input_image = self.read_image()
        self.hidden_state = self._encoder(self.input_image, trainable = self.trainable)
        self.pred_hidden_state = self._LSTM(tf.squeeze(self.hidden_state[0:self.batch_size/2,-1]),trainable = self.trainable)
        self.pred_loss = tf.reduce_sum(tf.squared_difference(self.pred_hidden_state,tf.expand_dims(self.hidden_state[self.batch_size/2:self.batch_size,-1], 1))) 
        self.generated_images = self._decoder(tf.concat([tf.expand_dims(self.hidden_state[0:self.batch_size/2,-1], 1) ,self.pred_hidden_state],0), trainable = self.trainable)
        self.hidden_contin_loss = tf.reduce_sum(tf.abs(self.hidden_state[:,0,0,0:510] + self.hidden_state[:,0,0,2:512] - 2*self.hidden_state[:,0,0,1:511]))
        self.l2_loss = tf.reduce_sum(tf.squared_difference(self.input_image,self.generated_images)) + self.hidden_contin_loss + self.pred_loss
        tf.summary.scalar("L2_loss", self.l2_loss)
        tf.summary.scalar("hidden_contin_loss", self.hidden_contin_loss)
        tf.summary.scalar("pref_loss", self.pred_loss)
        self.train_variables = tf.trainable_variables()
        global_step = tf.Variable(0, trainable=False)
        #self.lr = tf.train.exponential_decay(self.learning_rate,global_step=global_step,decay_steps=500,decay_rate=0.8)
        #tf.summary.scalar("learning_rate", self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1 = 0.9)
        self.grads = self.optimizer.compute_gradients(self.l2_loss, var_list=self.train_variables)
        self.auto_op = self.optimizer.apply_gradients(self.grads)
        #self.add_global_op = global_step.assign_add(1)
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
        for itr in range(self.iterations):
            _, l2_loss_eval,hidden_contin_loss_eval,pred_loss_eval = self.sess.run([self.auto_op,self.l2_loss,self.hidden_contin_loss,self.pred_loss])
            print str(itr)+' '+str(l2_loss_eval)+' '+str(hidden_contin_loss_eval)+' '+str(pred_loss_eval)
            if itr % 100 == 99:
                ori_images_eval, gen_images_eval = self.sess.run([self.input_image,self.generated_images])
                self._save_images(ori_images_eval,"/train/ori_"+str(0))
                self._save_images(gen_images_eval,"/train/gen_"+str(0))
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, itr)
            if itr % 100 == 99:
                self.saver.save(self.sess, self.logs_dir + "model.ckpt", global_step=itr)
            if itr % 1000 == 199:
                self.learning_rate *= 0.5
        coord.request_stop()
        coord.join(threads)
    def test(self):
        self.config = tf.ConfigProto()  
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        fout_ori = open('ori_hidden_state.txt','w')
        fout_pre = open('pre_hidden_state.txt','w')
        for itr in range(23):
            [hidden_state_eval,pred_hidden_state_eval,ori_images_eval,gen_images_eval] = self.sess.run([self.hidden_state,self.pred_hidden_state,self.input_image,self.generated_images])
            hidden_state_eval = np.array(hidden_state_eval)
            print np.shape(hidden_state_eval)
            for i in range(self.batch_size/2):
                for j in range(512):
                    fout_ori.write(str(hidden_state_eval[i][0][0][j])+' ')
                    fout_pre.write(str(hidden_state_eval[i][0][0][j])+' ')
                fout_ori.write('\n')
                fout_pre.write('\n')
            for i in range(self.batch_size/2):
                for j in range(512):
                    fout_ori.write(str(pred_hidden_state_eval[i+32][0][0][j])+' ')
                    fout_pre.write(str(pred_hidden_state_eval[i][0][0][j])+' ')
                fout_ori.write('\n')
                fout_pre.write('\n')
            #self._save_images(ori_images_eval,"/train/ori_"+str(itr))
            #self._save_images(gen_images_eval,"/train/gen_"+str(itr))
        fout_ori.close()
        fout_pre.close()
        coord.request_stop()
        coord.join(threads)

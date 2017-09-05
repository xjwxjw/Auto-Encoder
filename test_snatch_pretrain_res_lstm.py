from Auto_Res_LSTM_Snatch_Pretrain import *
def main(argv=None):
    model = AutoEncoder(True)
    model.construct_network()
    model.train()
    #model.test()
if __name__ == "__main__":
    tf.app.run()

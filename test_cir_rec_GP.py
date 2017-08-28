from Auto_Encoder_Cir_Rec_GP import *
def main(argv=None):
    model = AutoEncoder(True)
    model.construct_network()
    model.train()
    #model.test()
if __name__ == "__main__":
    tf.app.run()

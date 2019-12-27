import os
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
from ipu_utils import *
import argparse

from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope
from gc_profile import save_tf_report
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
parser.add_argument("--num_accelerators", type=int, default=1, help="number of accelerators used for training")
parser.add_argument("--gc_profile", type=str, default="disable", help="enable/disable gc-profile")
parser.add_argument("--platform", type=str, default="IPU", help="IPU, IPU_MODEL")
args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
TOTAL_TRAIN_SIZE = 512000
NUID = 543060
NMID = 367983
NCAT = 1601

def train(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 32,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        data_type = 'FP32',
	seed = 2,
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    print("data_type: ", data_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)

    if model_type == 'DIN':
        model = Model_DIN(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = Model_DIN_V2_Gru_att_Gru(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = Model_DIN_V2_Gru_Gru_att(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = Model_DIN_V2_Gru_QA_attGru(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = Model_DIN_V2_Gru_Vec_attGru(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIEN':
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    else:
        print ("Invalid model_type : %s"%model_type)

    with ipu_scope('/device:IPU:0'):
        model.build_input()
        batch = ipu_compiler.compile(model.build_train, [])

    with tf.Session() as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)

        # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"
        ipu_options = utils.create_ipu_config(profiling=True, profile_execution=True)
        ipu_options = utils.auto_select_ipus(ipu_options, [1])

        utils.configure_ipu_system(ipu_options)
        utils.move_variable_initialization_to_cpu()
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        # writer = tf.summary.FileWriter('./din_log')
        # writer.add_graph(sess.graph)

        if (args.gc_profile==True):
            with tf.device('cpu'):
                report = gen_ipu_ops.ipu_event_trace()

        iter = 0
        lr = 0.001
        train_size = 0
        approximate_accelerator_time = 0
        for itr in range(1):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, args.data_type, maxlen, return_neg=True)
                start_time = time.time()
                
                loss, acc, aux_loss = sess.run(batch, feed_dict = {
                    model.uid_batch_ph: uids,
                    model.mid_batch_ph: mids,
                    model.cat_batch_ph: cats,
                    model.mid_his_batch_ph: mid_his,
                    model.cat_his_batch_ph: cat_his,
                    model.mask: mid_mask,
                    model.target_ph: target,
                    model.seq_len_ph: sl,
                    model.lr: lr,
                    #model.noclk_mid_batch_ph: noclk_mids,
                    #model.noclk_cat_batch_ph: noclk_cats,
                    })
                end_time = time.time()
                # print("training time of one batch: %.3f" % (end_time - start_time))
                approximate_accelerator_time += end_time - start_time
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                train_size += batch_size
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    # print("train_size: %d" % train_size)
                    # print("approximate_accelerator_time: %.3f" % approximate_accelerator_time)
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))

                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))
                if train_size >= TOTAL_TRAIN_SIZE:
                    break
            lr *= 0.5

            if (args.gc_profile==True):
                out = sess.run(report)
                save_tf_report(out)

            if train_size >= TOTAL_TRAIN_SIZE:
                break

            break

        print("iter: %d" % iter)
        print("Total recommendations: %d" % TOTAL_TRAIN_SIZE)
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))


if __name__ == '__main__':
    SEED = args.seed
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'test':
        test(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    else:
        print('do nothing...')



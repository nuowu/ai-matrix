import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import argparse

import os

from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope

from gc_profile import save_tf_report
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
parser.add_argument("--max_len", type=int, default=100, help="max seq len")
parser.add_argument("--num_accelerators", type=int, default=1, help="number of accelerators used for training")
parser.add_argument('--profiling', default=False, help = "profiling")
args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

TOTAL_TRAIN_SIZE = 512000

NUID = 543060
NMID = 367983
NCAT = 1601


def prepare_data(input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    lengths_x_tmp = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_mid_tmp = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_mid_tmp = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        new_lengths_x_tmp = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_mid_tmp.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
                new_lengths_x_tmp.append(maxlen)
            else:
                list_1 = [0] * (maxlen - len(inp[3]))
                new_seqs_mid.append(inp[3] + list_1)

                list_2 = [0] * (maxlen - len(inp[4]))
                new_seqs_cat.append(inp[4] + list_2)

                list_3 = []
                for i in range(maxlen - len(inp[5])):
                    list_3.append([0, 0, 0, 0, 0])
                new_noclk_seqs_mid.append(inp[5] + list_3)

                list_4 = []
                for i in range(maxlen - len(inp[6])):
                    list_4.append([0, 0, 0, 0, 0])
                new_noclk_seqs_cat.append(inp[6] + list_4)

                new_seqs_mid_tmp.append(inp[3])

                new_lengths_x.append(maxlen)
                new_lengths_x_tmp.append(l_x)

        lengths_x = new_lengths_x
        lengths_x_tmp = new_lengths_x_tmp
        seqs_mid = new_seqs_mid
        seqs_mid_tmp = new_seqs_mid_tmp
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None


    # print(lengths_x)
    # print(lengths_x_tmp)

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    # print('maxlen_x is %4d' % maxlen_x)

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int32')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int32')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int32')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int32')

    if args.data_type == 'FP32':
        data_type = 'float32'
    elif args.data_type == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)

    # for idx, mask in enumerate(mid_mask):
    #     print(idx)
    #     print(mask)

    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        # mid_mask[idx, :lengths_x[idx]] = 1.
        mid_mask[idx, :lengths_x_tmp[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    # for idx, mask in enumerate(mid_mask):
    #     print(idx)
    #     print(mask)

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval(sess, test_data, model, model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0
    for src, tgt in test_data:
        nums += 1
        sys.stdout.flush()
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        # print("begin evaluation")
        start_time = time.time()
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
        end_time = time.time()
        # print("evaluation time of one batch: %.3f" % (end_time - start_time))
        # print("end evaluation")
        eval_time += end_time - start_time
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        # print("nums: ", nums)
        # break
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        if args.mode == 'train':
            model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums


def get_model(model_type, data_type, run_options):
    if model_type == 'DNN':
        return Model_DNN(run_options, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'PNN':
        return Model_PNN(run_options, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'Wide':
        return Model_WideDeep(run_options, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN':
        return Model_DIN(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-att-gru':
        return Model_DIN_V2_Gru_att_Gru(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-gru-att':
        return Model_DIN_V2_Gru_Gru_att(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        return Model_DIN_V2_Gru_QA_attGru(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        return Model_DIN_V2_Gru_Vec_attGru(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    elif model_type == 'DIEN':
        return Model_DIN_V2_Gru_Vec_attGru_Neg(run_options, NUID, NMID, NCAT, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
    else:
        sys.exit("Invalid model_type : %s" % model_type)


def train(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2,
        use_ipu=True,
):
    class RunningOptions:
        def __init__(self, batch_size = 4, sequence_length = 100, negative_samples = 5, max_iteration = 10):
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.negative_samples = negative_samples
            self.max_rnn_while_loops = max_iteration

    
    print("batch_size: ", batch_size)
    print("sequence lenght: ", maxlen)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    run_options = RunningOptions(batch_size = batch_size, sequence_length = maxlen)

    model = get_model(model_type, data_type, run_options)

    batch = None
    if use_ipu:
        with ipu_scope('/device:IPU:0'):
            model.build_input_ipu()
            batch = ipu_compiler.compile(model.build_train_ipu, [])

    if use_ipu:
        session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with session as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable dtype: ", var.dtype)
        #     if var.dtype == 'float32_ref':
        #         print("global variable: ", var)
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"
        if args.profiling:
            ipu_options = utils.create_ipu_config(profiling=True, profile_execution=True)
        else:
            ipu_options = utils.create_ipu_config(profiling=False, profile_execution=False)

        ipu_options = utils.auto_select_ipus(ipu_options, [1])

        utils.configure_ipu_system(ipu_options)
        utils.move_variable_initialization_to_cpu()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.profiling:
            with tf.device('cpu'):
                report = gen_ipu_ops.ipu_event_trace()

        sys.stdout.flush()
        # print('                                                                                      test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
        sys.stdout.flush()


        if args.profiling:
            lr = 0.0003
            train_data_tmp = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
            for src, tgt in train_data_tmp:
                if (model_type == 'DIEN'):
                    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
                    loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_mids, noclk_cats], ipu_output=batch)
                elif (model_type == 'DIN'):
                    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = prepare_data(src, tgt, maxlen, return_neg=False)
                    loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr], ipu_output=batch)

                with tf.Session() as sess:
                    summary = sess.run(report)
                save_tf_report(summary)
                break
        else:
            iter = 0
            # lr = 0.0006
            # lr = 0.002
            # lr = 0.01
            lr = 0.1
            approximate_accelerator_time = 0
            for itr in range(20):
                print ("iteration : %0d" % itr)
                train_size = 0
                loss_sum = 0.0
                accuracy_sum = 0.
                aux_loss_sum = 0.
                idx = 0
                train_data_tmp = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
                for src, tgt in train_data_tmp:
                    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)

                    start_time = time.time()
                    if (model_type == 'DIEN'):
                        loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats], ipu_output=batch)
                    elif (model_type == 'DIN'):
                        loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr], ipu_output=batch)
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
                        print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' %
                              (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                        loss_sum = 0.0
                        accuracy_sum = 0.0
                        aux_loss_sum = 0.0
                    # if (iter % save_iter) == 0:
                    #     print('save model iter: %d' % (iter))
                    #     model.save(sess, model_path+"--"+str(iter))

                    if train_size >= TOTAL_TRAIN_SIZE:
                        break

                lr *= 0.5

                # if train_size >= TOTAL_TRAIN_SIZE:
                #     break

            print("iter: %d" % iter)
            print("Total train_size : %d" % train_size)
            print("Total batch_size : %d" % batch_size)
            print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
            print("Approximate accelerator performance in recommendations/second is %.3f" % (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))


def test(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_best_model_trained/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print("global variable: ", var)
        if data_type == 'FP32':
            model.restore(sess, model_path)
        if data_type == 'FP16':
            fp32_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
            # print("fp32_variables: ", fp32_variables)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for variable in tf.global_variables():
                # print("variable: ", variable)
                if variable.op.name in fp32_variables:
                    var = tf.contrib.framework.load_variable(model_path, variable.op.name)
                    # print("var: ", var)
                    # print("var.dtype: ", var.dtype)
                    if(variable.dtype == 'float16_ref'):
                        tf.add_to_collection('assignOps', variable.assign(tf.cast(var, tf.float16)))
                        # print("var value: ", sess.run(tf.cast(var, tf.float16)))
                    else:
                        tf.add_to_collection('assignOps', variable.assign(var))
                else:
                    raise ValueError("Variable %s is missing from checkpoint!" % variable.op.name)
            sess.run(tf.get_collection('assignOps'))
            # for variable in sess.run(tf.get_collection('assignOps')):
            #     print("after load checkpoint: ", variable)
        # for variable in tf.global_variables():
        #     print("after load checkpoint: ", sess.run(variable))
        approximate_accelerator_time = 0
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        test_auc, test_loss, test_accuracy, test_aux_loss, eval_time, num_iters = eval(sess, test_data, model, model_path)
        approximate_accelerator_time += eval_time
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.9f ---- test_aux_loss: %.4f ---- eval_time: %.3f' % (test_auc, test_loss, test_accuracy, test_aux_loss, eval_time))
        print("Total recommendations: %d" % (5*num_iters*batch_size))
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(5*num_iters*batch_size)/float(approximate_accelerator_time)))

if __name__ == '__main__':
    SEED = args.seed
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type, maxlen=args.max_len)
    elif args.mode == 'test':
        test(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type, maxlen=args.max_len)
    else:
        print('do nothing...')

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import heapq

from model import Transformer
from load_data import *
import os, codecs
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

maxlen = 20
batch_size = 32
num_epochs = 30
hidden_units = 512
num_blocks = 6
num_heads = 8
learning_rate = 0.0001
dropout_rate = 0.1

source_vocab_file = './source_vocab.tsv'
target_vocab_file = './target_vocab.tsv'
logdir = './logdir'
train_file = '../data/train.txt'


def train(file_name):
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./backup/latest/'):
        os.mkdir('./backup/latest/')
    if not os.path.exists('./summaries'):
        os.mkdir('./summaries')
    # Load vocabulary    
    source_word_index, source_index_word = load_vocab(source_vocab_file)
    target_word_index, target_index_word = load_vocab(target_vocab_file)
    
    # Construct graph
    g = Transformer(
        maxlen = maxlen,
        batch_size = batch_size,
        source_vocab_size = len(source_word_index),
        target_vocab_size = len(target_word_index),
        hidden_units = hidden_units,
        num_blocks = num_blocks,
        num_heads = num_heads,
        dropout_rate = dropout_rate,
        learning_rate = learning_rate,
        file_name = file_name
    )
    
    with g.graph.as_default():
        summary_writer = tf.summary.FileWriter('./summaries/')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # 根据需要分配显存
        config.allow_soft_placement = True # 自动选择设备

        # Start session
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        #sess.run(tf.initialize_local_variables())

        if tf.train.get_checkpoint_state('./backup/latest/'):
            saver = tf.train.Saver()
            saver.restore(sess, './backup/latest/')
            print('********Restore the latest trained parameters.********')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for epoch in range(1, num_epochs+1): 
            if coord.should_stop():
                break
            print('epoch: {}/{}'.format(epoch, num_epochs))
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                gs = sess.run(g.global_step)   
                if gs % 1500 == 0:
                    summary = sess.run(g.merged)
                    summary_writer.add_summary(summary, gs)
                    saver = tf.train.Saver()
                    saver.save(sess, './backup/latest/', write_meta_graph=False)
        coord.request_stop()
        coord.join(threads)
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest/', write_meta_graph=False)
        sess.close()
        
        print("Done")    
    
def evaluate(in_file, out_file): 
    # Load data
    X, Y, sources, targets = load_data(in_file, maxlen)
    source_word_index, source_index_word = load_vocab(source_vocab_file)
    target_word_index, target_index_word = load_vocab(target_vocab_file)

    # Load graph
    g = Transformer(
        maxlen = maxlen,
        batch_size = 1,
        source_vocab_size = len(source_word_index),
        target_vocab_size = len(target_word_index),
        hidden_units = hidden_units,
        num_blocks = num_blocks,
        num_heads = num_heads,
        learning_rate = learning_rate,
        dropout_rate = dropout_rate,
        is_training = False
    )
    
     
    # Start session         
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with g.graph.as_default():    
        with tf.Session(config=config) as sess:
            ## Restore parameters
            if tf.train.get_checkpoint_state('./backup/latest/'):
                saver = tf.train.Saver()
                saver.restore(sess, './backup/latest/')
                print('********Restore the latest trained parameters.********')
            else:
                print('********The model is not existed.********')
              
            ## Inference
            with codecs.open(out_file, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X)):
                     
                    ### Get mini-batches
                    x = X[i: (i+1)]
                    source = sources[i: (i+1)]
                    target = targets[i: (i+1)]
                     
                    for max_idx in range(1, 3):
                        for k in range(-1, maxlen // 2):
                            ### Autoregressive inference
                            preds = np.zeros((1, maxlen), np.int32)
                            for j in range(maxlen):
                                _logits, _preds = sess.run([g.logits, g.preds], {g.x: x, g.y: preds})
                                _logits = _logits[0,j,:]
                                word_indices = heapq.nlargest(3, range(len(_logits)), _logits.take)
                                if j == k:
                                    preds[0, j] = word_indices[max_idx]
                                else:
                                    preds[:, j] = _preds[:, j]
                                #print(target_index_word[_preds[0,j]], target_index_word[preds[0,j]])
                                #print(target_index_word[word_indices[0]], target_index_word[word_indices[1]])
                             
                            ### Write to file
                            for s, t, pred in zip(source, target, preds): # sentence-wise
                                got = " ".join(target_index_word[idx] for idx in pred).split("</S>")[0].strip()
                                fout.write(got +"\t" + s + "\n")
                                #fout.write("- expected: " + t + "\n")
                                #fout.write("- got: " + got + "\n\n")
                                fout.flush()
                                  
                                # bleu score
                                #ref = t.split()
                                #hypothesis = got.split()
                                #if len(ref) > 3 and len(hypothesis) > 3:
                                #    list_of_refs.append([ref])
                                #    hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                #score = corpus_bleu(list_of_refs, hypotheses)
                #print("bleu_score = " + str(100*score))
                #fout.write("Bleu Score = " + str(100*score))

def eval():
    n_iter = 1
    in_file = '../data/test.txt'
    if not os.path.exists('results'): 
        os.mkdir('results')
    for i in range(1, n_iter+1):
        out_file = './results/out_' + str(i) + '.txt'
        print('iter_num: {}/{}'.format(i, n_iter))
        evaluate(in_file, out_file)
        in_file = out_file


if __name__ == '__main__':                
    #train(train_file)
    eval()

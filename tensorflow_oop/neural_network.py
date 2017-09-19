"""
Neural network class -- parent for all models.
"""

import tensorflow as tf
import numpy as np
import os
import time
import sys
from tensorflow.contrib.tensorboard.plugins import projector

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import *
from tensorflow_oop.decorators import *

# Set logging level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TFNeuralNetwork(object):

    """
    Basic neural network model.
    """

    __slots__ = ['init', 'loaded', 'log_dir',
                 'inputs_shape', 'targets_shape', 'outputs_shape',
                 'inputs', 'targets', 'outputs',
                 'top_k_placeholder', 'top_k_outputs',
                 'loss', 'global_step',
                 'sess',
                 'kwargs', 'metrics',
                 '_summary_writer', '_projector_config',
                 '_best_val_checkpoint', '_best_val_result',
                 '_best_val_iteration', '_best_val_key',
                 '_fit_checkpoint', '_vis_checkpoint',
                 '_iteration', '_iter_count',
                 '_epoch', '_epoch_count']

    def __init__(self, log_dir, reset=True):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.log_dir = log_dir
        self.init = False
        self.loaded = False
        self.metrics = {'batch_train': {},
                        'batch_validation': {},
                        'log_train': {},
                        'eval_train': {},
                        'eval_validation': {},
                        'eval_test': {}}

        # Checkpoint paths
        self._fit_checkpoint = os.path.join(self.log_dir, 'fit-checkpoint')
        self._vis_checkpoint = os.path.join(self.log_dir, 'vis-checkpoint')
        self._best_val_checkpoint = os.path.join(self.log_dir, 'best-val-checkpoint')

        # Reset default graph if necessary
        if reset:
            tf.reset_default_graph()

    def inference(self, inputs, **kwargs):
        """Model inference.

        Arguments:
            inputs -- tensor of batch with inputs
            kwargs -- dictionary of keyword arguments

        Return:
            outputs -- tensor of outputs layer

        """
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, targets, outputs, **kwargs):
        """Loss function.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of keyword arguments

        Return:
            loss -- tensorflow operation for minimization

        """
        raise Exception('Loss function should be overwritten!')
        return loss

    def load(self, model_checkpoint_path=None):
        """Load checkpoint.

        Arguments:
            model_checkpoint_path -- checkpoint path, search last if not passed

        """
        if model_checkpoint_path is None:
            model_checkpoint_path = tf.train.latest_checkpoint(self.log_dir)
            assert model_checkpoint_path is not None, \
                'Checkpoint path automatically not found.'

        print('Start loading model...')

        # Get metagraph saver
        saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta',
                                           clear_devices=True)

        # Create a session for running Ops on the Graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Restore model from saver
        saver.restore(self.sess, model_checkpoint_path)

        # Get named tensors
        self.inputs = self.sess.graph.get_tensor_by_name('inputs:0')
        self.targets = self.sess.graph.get_tensor_by_name('targets:0')
        self.outputs = self.sess.graph.get_tensor_by_name('outputs:0')
        self.top_k_placeholder = self.sess.graph.get_tensor_by_name('top_k_placeholder:0')
        self.top_k_outputs = self.sess.graph.get_tensor_by_name('top_k_outputs:0')
        self.loss = self.sess.graph.get_tensor_by_name('loss:0')
        self.global_step = self.sess.graph.get_tensor_by_name('global_step:0')

        def load_metrics(collection):
            collection_variables = self.sess.graph.get_collection(collection)
            collection_metrics = {}
            for var in collection_variables:
                name = var.name
                key = str(name[name.rfind('/') + 1 : name.rfind(':')])
                collection_metrics[key] = var
            return collection_metrics

        for collection in self.metrics:
            self.metrics[collection] = load_metrics('metric_' + collection)

        # Input, Target and Output layer shapes
        self.inputs_shape = self.inputs.shape.as_list()[1:]
        self.targets_shape = self.targets.shape.as_list()[1:]
        self.outputs_shape = self.outputs.shape.as_list()[1:]

        # Instantiate a SummaryWriter to output summaries and the Graph
        self._summary_writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)

        # Projector config object
        self._projector_config = projector.ProjectorConfig()

        # Enable initialization flag
        self.init = True
        self.loaded = True
        print('Model loaded from: %s' % model_checkpoint_path)

    def initialize(self,
                   inputs_shape,
                   targets_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   targets_type=tf.float32,
                   outputs_type=tf.float32,
                   clear_log_dir=False,
                   **kwargs):
        """Initialize model.

        Arguments:
            inputs_shape -- shape of inputs layer
            targets_shape -- shape of targets layer
            outputs_shape -- shape of outputs layer
            inputs_type -- type of inputs layer
            targets_type -- type of targets layer
            outputs_type -- type of outputs layer
            clear_log_dir -- indicator of clearing logging directory
            kwargs -- dictionary of keyword arguments

        """
        print('Start initializing model...')

        # Reset if necessary
        if clear_log_dir:
            # Clean TensorBoard logging directory
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)

        # Create TensorBoard logging directory
        if not tf.gfile.Exists(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)

        # Create a session for running Ops on the Graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Arguments
        self.kwargs = kwargs

        # Input, Target and Output layer shapes
        self.inputs_shape = list(inputs_shape)
        self.targets_shape = list(targets_shape)
        self.outputs_shape = list(outputs_shape)

        # Generate placeholders for the inputs and targets
        self.inputs = tf.placeholder(inputs_type,
                                     shape=[None] + self.inputs_shape,
                                     name='inputs')
        self.targets = tf.placeholder(targets_type,
                                      shape=[None] + self.targets_shape,
                                      name='targets')

        # Build a Graph that computes predictions from the inference model
        outputs = self.inference(self.inputs, **self.kwargs)
        self.outputs = tf.identity(outputs, name='outputs')

        # Top K outputs
        self.top_k_placeholder = tf.placeholder(tf.int32, [], name='top_k_placeholder')
        self.top_k_outputs = tf.nn.top_k(self.outputs,
                                         k=self.top_k_placeholder,
                                         name='top_k_outputs')

        # Loss function
        loss = self.loss_function(self.targets, self.outputs, **self.kwargs)
        self.loss = tf.identity(loss, name='loss')

        # Global step tensor
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add loss metric
        self.add_metric(self.loss,
                        summary_type=tf.summary.scalar,
                        collections=['batch_train',
                                     'batch_validation',
                                     'log_train'])

        # Instantiate a SummaryWriter to output summaries and the Graph
        self._summary_writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)

        # Projector config object
        self._projector_config = projector.ProjectorConfig()

        # Run the Op to initialize the variables
        self.initialize_variables()

        # Enable initialization flag
        self.init = True

        print('Finish initializing model.')

    def initialize_variables(self):
        """Initialize global and local variables."""
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def add_metric(self,
                   metric,
                   summary_type,
                   collections):
        """Add logging and summarizing metric.

        Arguments:
            metric -- tensorflow operation
            summary_type -- tensorflow summary type (e.g. tf.summary.scalar)
            collections -- list of strings from ['batch_train',
                                                 'batch_validation',
                                                 'log_train',
                                                 'eval_train',
                                                 'eval_validation',
                                                 'eval_test']

        """
        assert isinstance(metric, tf.Tensor), \
            '''Metric should be tf.Tensor:
            type(metric) = %s''' % type(metric)
        for collection in collections:
            assert collection in ['batch_train',
                                  'batch_validation',
                                  'log_train',
                                  'eval_train',
                                  'eval_validation',
                                  'eval_test'], \
                '''Collections should be only from list
                ['batch_train',
                 'batch_validation',
                 'log_train',
                 'eval_train',
                 'eval_validation',
                 'eval_test']:
                collection = %s''' % collection
        name = metric.name
        key = str(name[name.rfind('/') + 1 : name.rfind(':')])
        if key[-1].isdigit():
            while key[-1].isdigit():
                key = key[:-1]
            key = key[:-1]

        for collection in collections:
            summary_type(collection + '/' + key,
                         metric,
                         collections=[collection])
            self.sess.graph.add_to_collection('metric_' + collection, metric)
            self.metrics[collection][key] = metric
        self.sess.run(tf.local_variables_initializer())

    @check_initialization
    @check_fit_arguments
    def fit(self,
            train_set,
            epoch_count=None,
            iter_count=None,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            val_set=None,
            summarizing_period=100,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000,
            max_gradient_norm=None,
            best_val_key=None):
        """Train model.

        Arguments:
            train_set -- dataset for training
            epoch_count -- training epochs count
            iter_count -- training iterations count
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step or tensor
            val_set -- dataset for validation
            summarizing_period -- iterations count between summarizing
            logging_period -- iterations count between logging to stdout
            checkpoint_period -- iterations count between saving checkpoint
            evaluation_period -- iterations count between evaluation
            max_gradient_norm -- maximal gradient norm for clipping
            best_val_key -- metric key for saving best validation checkpoint

        """
        print('Start training...')
        start_fit_time = time.time()

        # Update actual iter and epoch count
        self._get_actual_iter_epoch_count(iter_count, epoch_count,
                                          train_set.size, train_set.batch_size)

        # Global iter step and epoch number
        self._iteration = self.global_step.eval(session=self.sess)
        self._epoch = self._iteration * train_set.batch_size // train_set.size
        assert self._iter_count >= self._iteration, \
            '''Iteration count should be greater than init iteration:
            self._iter_count = %s, self._iteration = %s''' \
            % (self._iter_count, self._iteration)
        if self._iter_count == self._iteration:
            print('Init iteration is equal to iteration count.')
            return

        # Get training operation
        train_op = self._get_train_op(optimizer, learning_rate, max_gradient_norm)

        # Print training options
        self._print_training_options(self._epoch_count,
                                     self._iter_count,
                                     optimizer,
                                     learning_rate,
                                     train_set.batch_size,
                                     val_set.batch_size if val_set else None,
                                     summarizing_period,
                                     logging_period,
                                     checkpoint_period,
                                     evaluation_period,
                                     max_gradient_norm,
                                     best_val_key)

        # Calculate initial result on validation set
        if val_set is not None and best_val_key is not None:
            print('Initial evaluation...')
            self._best_val_key = best_val_key
            self.evaluate_and_log(val_set.full_batch(), 'eval_validation')

        # Initial logging period time
        self._last_log_time = time.time()

        # Loop over all batches
        while self._iteration < self._iter_count:
            # Get current iteration and epoch
            self._iteration += 1
            self._epoch = self._iteration * train_set.batch_size // train_set.size

            # Calculate current iteration options
            last_iteration_flag = self._iteration == self._iter_count
            summarizing_flag = self._iteration % summarizing_period == 0 or last_iteration_flag
            logging_flag     = self._iteration % logging_period     == 0 or last_iteration_flag
            checkpoint_flag  = self._iteration % checkpoint_period  == 0 or last_iteration_flag
            evaluation_flag  = self._iteration % evaluation_period  == 0 or last_iteration_flag

            # One training iteration
            self._training_step(train_set, train_op, summarizing_flag)

            # One validation iteration
            if val_set is not None:
                self._validation_step(val_set, summarizing_flag)

            # Print training progress periodically
            if logging_flag:
                self.evaluate_and_log(train_set.last_batch, 'log_train')

            # Save a checkpoint the model periodically
            if checkpoint_flag:
                print('Saving checkpoint periodically...')
                self.save(self._fit_checkpoint, global_step=self._iteration)

            # Evaluate the model periodically
            if evaluation_flag:
                print('Evaluation...')
                self.evaluate_and_log(train_set.full_batch(), 'eval_train')

                # Eval on validation set if necessary
                if val_set is not None:
                    self.evaluate_and_log(val_set.full_batch(), 'eval_validation')

        self._summary_writer.flush()
        total_time = time.time() - start_fit_time
        print('Finish training iteration (total time %.3f sec).\n' % total_time)

        if val_set is not None and best_val_key is not None:
            print('Report by best result on validation set:')
            print('%20s : %s' % ('best_val_key', self._best_val_key))
            print('%20s : %s' % ('best_val_result', self._best_val_result))
            print('%20s : %s' % ('best_val_iteration', self._best_val_iteration))
            print('%20s : %s\n' % ('best_val_checkpoint', self._best_val_checkpoint))

    @check_initialization
    def fill_feed_dict(self, batch):
        """Get filled feed dictionary for batch.

        Arguments:
            batch -- batch of inputs

        """
        feed_dict = {
            self.inputs: batch.data,
            self.targets: batch.labels,
        }
        return feed_dict

    @check_initialization
    @check_produce_arguments
    def produce(self, dataset, batch, output_tensors):
        """Produce model on batch and return list of output tensors values.

        Arguments:
            dataset -- TFDataset object
            batch -- TFBatch object
            output_tensors -- list of output tensors

        Return:
            output_values -- list of output tensors values

        """

        # Fill feed dict
        feed_dict = self.fill_feed_dict(batch)

        # Run one step of the model
        output_values = self.sess.run(output_tensors, feed_dict=feed_dict)

        return output_values

    @check_initialization
    @check_evaluate_arguments
    def evaluate(self, batch, collection='eval_test'):
        """Evaluate model.

        Arguments:
            batch -- TFBatch object
            collection -- string value from ['batch_train',
                                             'batch_validation',
                                             'log_train',
                                             'eval_train',
                                             'eval_validation',
                                             'eval_test']

        Return:
            metrics -- dictionary

        """

        # Init output values
        metrics = {}

        # Check if nothing to do
        if len(self.metrics[collection]) > 0:
            # Get key and tensors
            metric_keys = list(self.metrics[collection].keys())
            metric_tensors = list(self.metrics[collection].values())
            
            # Calculate metrics values
            feed_dict = self.fill_feed_dict(batch)
            metric_values = self.sess.run(metric_tensors, feed_dict=feed_dict)
            for i in range(len(metric_keys)):
                metrics[metric_keys[i]] = metric_values[i]

            # Update the events file with evaluation summary
            if collection != 'log_train':
                summary_str = self.sess.run(tf.summary.merge_all(collection),
                                            feed_dict=feed_dict)
                self._summary_writer.add_summary(summary_str, self._iteration)

            # Save best result on validation set if necessary
            if self._best_val_key is not None and collection == 'eval_validation':
                result = metrics[self._best_val_key]
                if self._best_val_result is None or result > self._best_val_result:
                    print('Saving checkpoint with best result on validation set...')
                    self.save_best_on_validation(result)

        return metrics

    @check_initialization
    @check_evaluate_arguments
    def evaluate_and_log(self, batch, collection='eval_test'):
        """Evaluate model.

        Arguments:
            batch -- TFBatch object
            collection -- string value from ['batch_train',
                                             'batch_validation',
                                             'log_train',
                                             'eval_train',
                                             'eval_validation',
                                             'eval_test']

        Return:
            metrics -- dictionary

        """

        # Evaluate on current collection
        start_evaluation_time = time.time()
        metrics = self.evaluate(batch, collection)
        duration = time.time() - start_evaluation_time

        # Convert metrics to string
        metrics_str = '   '.join(['%s = %.6f' % (k, metrics[k]) for k in metrics])

        if collection == 'log_train':
            # Calculate time of last logging period
            period_time = time.time() - self._last_log_time
            self._last_log_time = time.time()

            # Log training process
            format_string = 'Iter %d / %d (epoch %d / %d):   %s   [%.3f sec]'
            print(format_string % (self._iteration,
                                   self._iter_count,
                                   self._epoch,
                                   self._epoch_count,
                                   metrics_str,
                                   period_time))
        elif len(metrics) > 0:
            # Log evaluation result
            format_string = 'Evaluation on [%s]:   %s   [%.3f sec]'
            print(format_string % (collection, metrics_str, duration))
        return metrics

    @check_initialization
    def save(self, filename, global_step=None):
        """Save checkpoint.

        Arguments:
            filename -- path to saving
            global_step -- optional suffix adding to path (default None)

        """
        saver = tf.train.Saver(max_to_keep=None)
        saved_filename = saver.save(self.sess, filename,
                                    global_step=global_step)
        print('Model saved to: %s' % saved_filename)
    
    @check_initialization
    def restore(self, filename):
        """Restore checkpoint only if model initialized.
        
        Arguments:
            filename -- path to checkpoint
        
        """
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(self.sess, filename)

    @check_initialization
    def save_best_on_validation(self, result):
        """Save checkpoint with best result on validation set.

        Arguments:
            result -- new best validation result

        """
        self._best_val_result = result
        self._best_val_iteration = self._iteration
        self.save(self._best_val_checkpoint)

    @check_initialization
    def restore_best_on_validation(self):
        """Restore checkpoint with best result on validation set."""
        self.restore(self._best_val_checkpoint)

    @check_initialization
    @check_inputs_values
    def forward(self, inputs_values):
        """Forward propagation.

        Arguments:
            inputs_values -- batch of inputs

        Return:
            outputs_values -- batch of outputs

        """
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs_values,
        })

    @check_initialization
    @check_inputs_values
    def top_k(self, inputs_values, k):
        """Top k outputs.

        Arguments:
            inputs_values -- batch of inputs
            k -- top outputs count

        Return:
            top_k_values -- batch of top k outputs

        """
        return self.sess.run(self.top_k_outputs, feed_dict={
            self.inputs: inputs_values,
            self.top_k_placeholder: k
        })

    def __str__(self):
        string = 'TFNeuralNetwork object:\n'
        for attr in self.__slots__:
            if hasattr(self, attr) and attr[0] != '_':
                if attr == 'metrics':
                    buf = ''
                    collections = sorted(list(self.metrics.keys()))
                    for collection in collections:
                        keys = list(self.metrics[collection].keys())
                        buf += '%30s: %s\n' % (collection, sorted(keys))
                    string += '%20s:\n%s' % (attr, buf)
                elif attr == 'kwargs':
                    buf = ''
                    keys = sorted(list(self.kwargs.keys()))
                    for key in keys:
                        buf += '%30s: %s\n' % (key, self.kwargs[key])
                    string += '%20s:\n%s' % (attr, buf)
                else:
                    string += '%20s: %s\n' % (attr, getattr(self, attr))
        return string[:-1]

    def _get_actual_iter_epoch_count(self, iter_count, epoch_count, dataset_size, batch_size):
        """Actualize iteration and epoch count.

        Arguments:
            iter_count -- iteration count (None if unknown)
            epoch_count -- epoch count (None if unknown)
            dataset_size -- length of dataset
            batch_size -- length of batch

        """
        if epoch_count is not None:
            iter_count_by_epoch = (dataset_size * epoch_count) // batch_size
            if dataset_size % batch_size != 0:
                iter_count_by_epoch += 1
            if iter_count is not None:
                iter_count = min(iter_count, iter_count_by_epoch)
            else:
                iter_count = iter_count_by_epoch
        else:
            epoch_count = (iter_count * batch_size) // dataset_size
        self._iter_count = iter_count
        self._epoch_count = epoch_count

    def _get_train_op(self, optimizer, learning_rate, max_gradient_norm):
        """Get training operation.

        Arguments:
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step
            max_gradient_norm -- maximal gradient norm for clipping

        Return:
            trin_op -- training operation

        """
        if not self.loaded:
            optimizer_op = optimizer(learning_rate)

            # Calculate gradients
            tvars = tf.trainable_variables()
            gradients = tf.gradients(self.loss, tvars)

            # Add tvars metric
            flatten_tvars = []
            for tvar in tvars:
                flatten_tvars.append(tf.reshape(tvar, [-1,]))
            concat_tvars = tf.concat(flatten_tvars, 0,
                                     name='all_tvars')
            self.add_metric(concat_tvars,
                            summary_type=tf.summary.histogram,
                            collections=['batch_train'])

            # Add gradients metric
            flatten_gradients = []
            for gradient in gradients:
                flatten_gradients.append(tf.reshape(gradient, [-1,]))
            concat_gradients = tf.concat(flatten_gradients, 0,
                                         name='all_gradients')
            self.add_metric(concat_gradients,
                            summary_type=tf.summary.histogram,
                            collections=['batch_train'])

            # Gradient clipping if necessary
            if max_gradient_norm is not None:
                # Calculate clipping gradients
                clip_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 
                                                                       max_gradient_norm)
                
                # Add clipping gradients metric
                flatten_clip_gradients = []
                for clip_gradient in clip_gradients:
                    flatten_clip_gradients.append(tf.reshape(clip_gradient, [-1,]))
                concat_clip_gradients = tf.concat(flatten_clip_gradients, 0,
                                                  name='all_clip_gradients')
                self.add_metric(concat_clip_gradients,
                                summary_type=tf.summary.histogram,
                                collections=['batch_train'])
                self.add_metric(tf.identity(gradient_norm, 'gradient_norm'),
                                summary_type=tf.summary.scalar,
                                collections=['batch_train'])

                # Add to the Graph the Ops that apply gradients
                train_op = optimizer_op.apply_gradients(zip(clip_gradients, tvars),
                                                        global_step=self.global_step,
                                                        name='train_op')
            else:
                # Add to the Graph the Ops that minimize loss
                train_op = optimizer_op.minimize(self.loss,
                                                 global_step=self.global_step,
                                                 name='train_op')

            # Run the Op to initialize the variables
            self.initialize_variables()
        else:
            train_op = self.sess.graph.get_operation_by_name('train_op')
        return train_op

    def _print_training_options(self,
                                epoch_count,
                                iter_count,
                                optimizer,
                                learning_rate,
                                train_batch_size,
                                val_batch_size,
                                summarizing_period,
                                logging_period,
                                checkpoint_period,
                                evaluation_period,
                                max_gradient_norm,
                                best_val_key):
        """Formatted print training options."""
        if epoch_count:
            print('%20s: %s' % ('epoch_count', epoch_count))
        print('%20s: %s' % ('iter_count', iter_count))
        print('%20s: %s' % ('optimizer', optimizer))
        print('%20s: %s' % ('learning_rate', learning_rate))
        print('%20s: %s' % ('train_batch_size', train_batch_size))
        if val_batch_size:
            print('%20s: %s' % ('val_batch_size', val_batch_size))
        print('%20s: %s' % ('summarizing_period', summarizing_period))
        print('%20s: %s' % ('logging_period', logging_period))
        print('%20s: %s' % ('checkpoint_period', checkpoint_period))
        print('%20s: %s' % ('evaluation_period', evaluation_period))
        if max_gradient_norm is not None:
            print('%20s: %s' % ('max_gradient_norm', max_gradient_norm))
        if best_val_key is not None:
            print('%20s: %s' % ('best_val_key', best_val_key))
        buf = ''
        collections = sorted(list(self.metrics.keys()))
        for collection in collections:
            keys = list(self.metrics[collection].keys())
            buf += '%30s: %s\n' % (collection, sorted(keys))
        print('%20s:\n%s' % ('metrics', buf))

    def _training_step(self, train_set, train_op, summarizing_flag):
        """Run one training iteration.

        Arguments:
            train_set -- TFDataset object
            train_op -- training operation
            summarizing_flag -- boolean indicator of summarizing

        """

        # Get next training batch
        train_batch = train_set.next_batch()

        # Produce this dataset over network
        if summarizing_flag and len(self.metrics['batch_train']) > 0:
            summary_op = tf.summary.merge_all('batch_train')
            values = self.produce(train_set, train_batch, [summary_op, train_op])
            summary_str = values[0]
            self._summary_writer.add_summary(summary_str, self._iteration)
        else:
            self.produce(train_set, train_batch, [train_op])

    def _validation_step(self, val_set, summarizing_flag):
        """Run one training iteration.

        Arguments:
            val_set -- TFDataset object
            summarizing_flag -- boolean indicator of summarizing

        """

        if summarizing_flag and len(self.metrics['batch_validation']) > 0:
            # Get next validation batch
            val_batch = val_set.next_batch()

            # Produce this dataset over network
            summary_op = tf.summary.merge_all('batch_validation')
            values = self.produce(val_set, val_batch, [summary_op])
            summary_str = values[0]
            self._summary_writer.add_summary(summary_str, self._iteration)
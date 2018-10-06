from itertools import islice
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

from vae.models import VariationalAutoEncoderModel

class VariationalAutoEncoder(object):
    def __init__(self, n_input_units, n_hidden_layers, n_hidden_units, n_latent_units,
                 learning_rate=0.05, batch_size=100, min_beta=1.0, max_beta=1.0,
                 distribution='normal', serial_layering=None):
        self.n_input_units = n_input_units
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.n_latent_units = n_latent_units
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.distribution = distribution
        if serial_layering:
            if not isinstance(serial_layering, (list, tuple)):
                raise TypeError("Argument 'serial_layering' must be a list or tuple of integers.")
            elif not all([isinstance(x, int) for x in serial_layering]):
                raise TypeError("Argument 'serial_layering' must be a list or tuple of integers.")
            elif sum(serial_layering) != self.n_hidden_layers:
                raise ValueError("Groupings in 'serial_layering' must sum to 'n_hidden_layers'.")
        self.serial_layering = serial_layering or [self.n_hidden_layers]
        self.layer_sequence = [sum(self.serial_layering[:i + 1]) for i in range(len(self.serial_layering))]

    class Encoder(object):
        def __init__(self, n_hidden_layers, n_hidden_units, n_latent_units, distribution, initializers=None):
            self.n_hidden_layers = n_hidden_layers
            self.n_hidden_units = n_hidden_units
            self.n_latent_units = n_latent_units
            self.distribution = distribution
            self.initializers = initializers

        def init_hidden_layers(self):
            self.hidden_layers = []
            self.applied_hidden_layers = []

        def add_hidden_layer(self, inputs):
            if self.initializers and self.initializers.get('layers', None):
                print("initializing encoder layer...")
                kernel_initializer, bias_initializer = self.initializers['layers'].pop(0)
            else:
                kernel_initializer, bias_initializer = None, None

            self.hidden_layers.append(tf.layers.Dense(units=self.n_hidden_units, activation=tf.nn.sigmoid,
                                                      kernel_initializer=kernel_initializer,
                                                      bias_initializer=bias_initializer))
            self.applied_hidden_layers.append(self.hidden_layers[-1].apply(inputs))
            return self.applied_hidden_layers[-1]

        def add_mu(self, inputs):
            if self.initializers and self.initializers.get('mu', None):
                print("initializing encoder mu...")
                kernel_initializer, bias_initializer = self.initializers['mu']
            else:
                kernel_initializer, bias_initializer = None, None

            if self.distribution == 'normal':
                self.mu = tf.layers.Dense(units=self.n_latent_units, kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer)
            elif self.distribution == 'vmf':
                self.mu = tf.layers.Dense(units=self.n_latent_units + 1,
                                          activation=lambda x: tf.nn.l2_normalize(x, axis=-1),
                                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
            else:
                raise NotImplemented

            self.applied_mu = self.mu.apply(inputs)
            return self.applied_mu

        def add_sigma(self, inputs):
            if self.initializers and self.initializers.get('sigma', None):
                print("initializing encoder sigma...")
                kernel_initializer, bias_initializer = self.initializers['sigma']
            else:
                kernel_initializer, bias_initializer = None, None

            if self.distribution == 'normal':
                self.sigma = tf.layers.Dense(units=self.n_latent_units,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                self.applied_sigma = self.sigma.apply(inputs)
            elif self.distribution == 'vmf':
                self.sigma = tf.layers.Dense(units=1, activation=tf.nn.softplus,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                self.applied_sigma = self.sigma.apply(inputs) + 1
            else:
                raise NotImplemented
            return self.applied_sigma

        def build(self, inputs):
            self.init_hidden_layers()

            layer = self.add_hidden_layer(inputs)

            for i in range(self.n_hidden_layers - 1):
                layer = self.add_hidden_layer(layer)

            mu = self.add_mu(layer)
            sigma = self.add_sigma(layer)

            return mu, sigma

        def eval(self, sess):
            layers = [
                sess.run([l.kernel, l.bias])
                for l in self.hidden_layers
            ]

            mu = sess.run([self.mu.kernel, self.mu.bias])

            sigma = sess.run([self.sigma.kernel, self.sigma.bias])

            return layers, mu, sigma

    class Decoder(object):
        def __init__(self, n_hidden_layers, n_hidden_units, n_output_units, initializers=None):
            self.n_hidden_layers = n_hidden_layers
            self.n_hidden_units = n_hidden_units
            self.n_output_units = n_output_units
            self.initializers = initializers

        def init_hidden_layers(self):
            self.hidden_layers = []
            self.applied_hidden_layers = []

        def add_hidden_layer(self, inputs):
            if self.initializers and self.initializers.get('layers', None):
                print("initializing decoder layer...")
                kernel_initializer, bias_initializer = self.initializers['layers'].pop(0)
            else:
                kernel_initializer, bias_initializer = None, None

            self.hidden_layers.append(tf.layers.Dense(units=self.n_hidden_units, activation=tf.nn.sigmoid,
                                                      kernel_initializer=kernel_initializer,
                                                      bias_initializer=bias_initializer))
            self.applied_hidden_layers.append(self.hidden_layers[-1].apply(inputs))
            return self.applied_hidden_layers[-1]

        def add_output(self, inputs):
            if self.initializers and self.initializers.get('output', None):
                print("initializing decoder output...")
                kernel_initializer, bias_initializer = self.initializers['output']
            else:
                kernel_initializer, bias_initializer = None, None

            self.output = tf.layers.Dense(units=self.n_output_units,
                                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
            self.applied_output = self.output.apply(inputs)
            return self.applied_output

        def build(self, inputs):
            self.init_hidden_layers()

            layer = self.add_hidden_layer(inputs)

            for i in range(self.n_hidden_layers - 1):
                layer = self.add_hidden_layer(layer)

            output = self.add_output(layer)

            return output

        def eval(self, sess):
            layers = [
                sess.run([l.kernel, l.bias])
                for l in self.hidden_layers
            ]

            output = sess.run([self.output.kernel, self.output.bias])

            return layers, output

    def sampled_z(self, mu, sigma, batch_size):
        if self.distribution == 'normal':
            epsilon = tf.random_normal(tf.stack([int(batch_size), self.n_latent_units]))
            z = mu + tf.multiply(epsilon, tf.exp(0.5 * sigma))
            loss = tf.reduce_mean(-0.5 * self.beta * tf.reduce_sum(1.0 + sigma - tf.square(mu) - tf.exp(sigma), 1))
        elif self.distribution == 'vmf':
            self.q_z = VonMisesFisher(mu, sigma, validate_args=True, allow_nan_stats=False)
            z = self.q_z.sample()
            self.p_z = HypersphericalUniform(self.n_latent_units, validate_args=True, allow_nan_stats=False)
            loss = tf.reduce_mean(-self.q_z.kl_divergence(self.p_z))
        else:
            raise NotImplemented

        return z, loss

    def build_feature_loss(self, x, output):
        return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, output), 1))

    def build_encoder_initializers(self, sess, n_hidden_layers):
        if hasattr(self, 'encoder'):
            result = {'layers': []}
            layers, mu, sigma = self.encoder.eval(sess)
            for i in range(n_hidden_layers):
                if layers:
                    kernel, bias = layers.pop(0)
                    result['layers'].append((tf.constant_initializer(kernel), tf.constant_initializer(bias)))
                else:
                    result['layers'].append((
                        tf.constant_initializer(np.diag(np.ones(self.n_latent_units))),
                        tf.constant_initializer(np.diag(np.ones(1)))
                    ))

            result['mu'] = (tf.constant_initializer(mu[0]), tf.constant_initializer(mu[1]))
            result['sigma'] = (tf.constant_initializer(sigma[0]), tf.constant_initializer(sigma[1]))
        else:
            result = None

        return result

    def build_decoder_initializers(self, sess, n_hidden_layers):
        if hasattr(self, 'decoder'):
            result = {'layers': []}
            layers, output = self.decoder.eval(sess)
            for i in range(n_hidden_layers):
                if layers:
                    kernel, bias = layers.pop(0)
                    result['layers'].append((tf.constant_initializer(kernel), tf.constant_initializer(bias)))
                else:
                    result['layers'].append((
                        tf.constant_initializer(np.diag(np.ones(self.n_latent_units))),
                        tf.constant_initializer(np.diag(np.ones(1)))
                    ))

            result['output'] = (tf.constant_initializer(output[0]), tf.constant_initializer(output[1]))
        else:
            result = None

        return result

    def build_initializers(self, attr_name, sess, n_hidden_layers):
        if hasattr(self, attr_name):
            layers = getattr(self, attr_name).eval(sess)[0]
            result = []
            for i in range(n_hidden_layers):
                if layers:
                    kernel, bias = layers.pop(0)
                    result.append((tf.constant_initializer(kernel), tf.constant_initializer(bias)))
                else:
                    result.append((
                        tf.constant_initializer(np.diag(np.ones(self.n_latent_units))),
                        tf.constant_initializer(np.diag(np.ones(1)))
                    ))
            return result
        else:
            return None

    def initialize_tensors(self, sess, n_hidden_layers=None):
        n_hidden_layers = n_hidden_layers or self.n_hidden_layers

        self.x = tf.placeholder("float32", [self.batch_size, self.n_input_units])
        self.beta = tf.placeholder("float32", [1, 1])
        self.encoder = self.Encoder(n_hidden_layers, self.n_hidden_units, self.n_latent_units, self.distribution,
                                    initializers=self.build_encoder_initializers(sess, n_hidden_layers))
        mu, sigma = self.encoder.build(self.x)
        self.mu = mu
        self.sigma = sigma

        z, latent_loss = self.sampled_z(self.mu, self.sigma, self.batch_size)
        self.z = z
        self.latent_loss = latent_loss

        self.decoder = self.Decoder(n_hidden_layers, self.n_hidden_units, self.n_input_units,
                                    initializers=self.build_decoder_initializers(sess, n_hidden_layers))
        self.output = self.decoder.build(self.z)

        self.feature_loss = self.build_feature_loss(self.x, self.output)
        self.loss = self.feature_loss + self.latent_loss

    def total_steps(self, data_count, epochs):
        num_batches = int(data_count / self.batch_size)
        return (num_batches * epochs) - epochs

    def generate_beta_values(self, total_steps):
        beta_delta = self.max_beta - self.min_beta
        log_beta_step = 5 / float(total_steps)
        beta_values = [
            self.min_beta + (beta_delta * (1 - math.exp(-5 + (i * log_beta_step))))
            for i in range(total_steps)
        ]
        return beta_values

    def train_from_rdd(self, data_rdd, epochs=1):
        data_count = data_rdd.count()
        total_steps = self.total_steps(data_count, epochs)
        beta_values = self.generate_beta_values(total_steps)

        layer_sequence_step = int(total_steps / len(self.layer_sequence))
        layer_sequence = self.layer_sequence.copy()

        with tf.Session() as sess:
            batch_index = 0
            for epoch_index in range(epochs):
                iterator = data_rdd.toLocalIterator()
                while True:
                    if (not batch_index % layer_sequence_step) and layer_sequence:
                        n_hidden_layers = layer_sequence.pop(0)
                        self.initialize_tensors(sess, n_hidden_layers)
                        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                        sess.run(tf.global_variables_initializer())

                    batch = np.array(list(islice(iterator, self.batch_size)))
                    if batch.shape[0] == self.batch_size:
                        beta = beta_values.pop(0) if len(beta_values) > 0 else self.min_beta
                        feed_dict = {self.x: np.array(batch), self.beta: np.array([[beta]])}

                        if not batch_index % 1000:
                            print("beta: {}".format(beta))
                            print("number of hidden layers: {}".format(n_hidden_layers))
                            ls, f_ls, d_ls = sess.run([self.loss, self.feature_loss, self.latent_loss],
                                                      feed_dict=feed_dict)
                            print("loss={}, avg_feature_loss={}, avg_latent_loss={}".format(ls, np.mean(f_ls),
                                                                                            np.mean(d_ls)))
                            print('running batch {} (epoch {})'.format(batch_index, epoch_index))
                        sess.run(optimizer, feed_dict=feed_dict)
                        batch_index += 1
                    else:
                        print("incomplete batch: {}".format(batch.shape))
                        break

            print("evaluating model...")
            encoder_layers, eval_mu, eval_sigma = self.encoder.eval(sess)
            decoder_layers, eval_output = self.decoder.eval(sess)

        return VariationalAutoEncoderModel(encoder_layers, eval_mu, eval_sigma, decoder_layers, eval_output)

    def train(self, data, visualize=False, epochs=1):
        data_size = data.shape[0]
        batch_size = self.batch_size
        total_steps = self.total_steps(data_size, epochs)
        beta_values = self.generate_beta_values(total_steps)

        layer_sequence_step = int(total_steps / len(self.layer_sequence))
        layer_sequence = self.layer_sequence.copy()

        with tf.Session() as sess:
            for epoch_index in range(epochs):
                i = 0
                while (i * batch_size) < data_size:
                    if (not i % layer_sequence_step) and layer_sequence:
                        n_hidden_layers = layer_sequence.pop(0)
                        self.initialize_tensors(sess, n_hidden_layers)
                        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                        sess.run(tf.global_variables_initializer())

                    batch = data[i * batch_size:(i + 1) * batch_size]
                    beta = beta_values.pop(0) if len(beta_values) > 0 else self.min_beta
                    feed_dict = {self.x: batch, self.beta: np.array([[beta]])}
                    sess.run(optimizer, feed_dict=feed_dict)
                    if visualize and (
                            not i % int((data_size / batch_size) / 3) or i == int(data_size / batch_size) - 1):
                        ls, d, f_ls, d_ls = sess.run([self.loss, self.output, self.feature_loss, self.latent_loss],
                                                     feed_dict=feed_dict)
                        plt.scatter(batch[:, 0], batch[:, 1])
                        plt.show()
                        plt.scatter(d[:, 0], d[:, 1])
                        plt.show()
                        print(i, ls, np.mean(f_ls), np.mean(d_ls))

                    i += 1

            encoder_layers, eval_mu, eval_sigma = self.encoder.eval(sess)
            decoder_layers, eval_output = self.decoder.eval(sess)

        return VariationalAutoEncoderModel(encoder_layers, eval_mu, eval_sigma, decoder_layers, eval_output)

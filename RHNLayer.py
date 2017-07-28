class Model(object):
  """A Variational RHN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.depth = depth = config.depth
    self.size = size = config.hidden_size
    self.num_layers = num_layers = config.num_layers
    vocab_size = config.vocab_size
    if vocab_size < self.size and not config.tied:
      in_size = vocab_size
    else:
      in_size = self.size
    self.in_size = in_size
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._noise_x = tf.placeholder(tf.float32, [batch_size, num_steps, 1])
    self._noise_i = tf.placeholder(tf.float32, [batch_size, in_size, num_layers])
    self._noise_h = tf.placeholder(tf.float32, [batch_size, size, num_layers])
    self._noise_o = tf.placeholder(tf.float32, [batch_size, 1, size])

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, in_size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data) * self._noise_x

    outputs = []
    self._initial_state = [0] * self.num_layers
    state = [0] * self.num_layers
    self._final_state = [0] * self.num_layers
    for l in range(config.num_layers):
      with tf.variable_scope('RHN' + str(l)):
        cell = RHNCell(size, in_size, is_training, depth=depth, forget_bias=config.init_bias)
        self._initial_state[l] = cell.zero_state(batch_size, tf.float32)
        state[l] = [self._initial_state[l], self._noise_i[:, :, l], self._noise_h[:, :, l]]
        for time_step in range(num_steps):
          if time_step > 0:
            tf.get_variable_scope().reuse_variables()
          (cell_output, state[l]) = cell(inputs[:, time_step, :], state[l])
          outputs.append(cell_output)
        inputs = tf.pack(outputs, axis=1)
        outputs = []

    output = tf.reshape(inputs * self._noise_o, [-1, size])


class RHNCell(RNNCell):
  """Variational Recurrent Highway Layer
  Reference: https://arxiv.org/abs/1607.03474
  """

  def __init__(self, num_units, in_size, is_training, depth=3, forget_bias=None):
    self._num_units = num_units
    self._in_size = in_size
    self.is_training = is_training
    self.depth = depth
    self.forget_bias = forget_bias

  @property
  def input_size(self):
    return self._in_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    current_state = state[0]
    noise_i = state[1]
    noise_h = state[2]
    for i in range(self.depth):
      with tf.variable_scope('h_'+str(i)):
        if i == 0:
          h = tf.tanh(linear([inputs * noise_i, current_state * noise_h], self._num_units, True))
        else:
          h = tf.tanh(linear([current_state * noise_h], self._num_units, True))
      with tf.variable_scope('t_'+str(i)):
        if i == 0:
          t = tf.sigmoid(linear([inputs * noise_i, current_state * noise_h], self._num_units, True, self.forget_bias))
        else:
          t = tf.sigmoid(linear([current_state * noise_h], self._num_units, True, self.forget_bias))
      current_state = (h - current_state)* t + current_state

    return current_state, [current_state, noise_i, noise_h]


def linear(args, output_size, bias, bias_start=None, scope=None):
  """
  This is a slightly modified version of _linear used by Tensorflow rnn.
  The only change is that we have allowed bias_start=None.
  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    elif bias_start is None:
      bias_term = vs.get_variable("Bias", [output_size], dtype=dtype)
    else:
      bias_term = vs.get_variable("Bias", [output_size], dtype=dtype,
                                  initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term
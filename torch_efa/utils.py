import torch, math

def _extract_max_degree_and_check_shape(shape):
  """Extract max_degree from feature shape and check for valid sizes.

  Valid feature shapes are at least three-dimensional:
  (..., 1, (max_degree+1)**2, num_features)  for SO(3) features, and
  (..., 2, (max_degree+1)**2, num_features)  for O(3) features, with max_degree
  being any positive integer or 0.

  Args:
    shape: The input shape to be checked.

  Returns:
    The value of `max_degree` extracted from shape.

  Raises:
    ValueError: If the shape is invalid.
  """

  # Check that shape has parity, degree, and feature channels.
  if len(shape) < 3:
    raise ValueError(
        f'shape of features must have at least length 3, received shape {shape}'
    )

  # Check that axis -3 (parity channels) has the correct size (1 or 2).
  if not (shape[-3] == 1 or shape[-3] == 2):
    raise ValueError(
        f'expected 1 or 2 for axis -3 of feature shape, received shape{shape}'
    )

  # Extract max_degree from size of axis -2 (degree channel).
  max_degree = round(math.sqrt(shape[-2]) - 1)

  # Check that axis -2 (degree channel) has a valid size.
  expected_size = (max_degree + 1) ** 2
  if shape[-2] != expected_size:
    raise ValueError(
        f'received invalid size {shape[-2]} for axis -2 of '
        f'feature shape, closest valid size is {expected_size}'
    )

  return max_degree

def change_max_degree_or_type(x, max_degree=None, include_pseudotensors=None):
  r"""Changes the maximum degree and/or type of features.

  When changing the maximum degree to a larger value, the features are padded
  with zeros to give them the correct shape. When changing the maximum degree to
  a smaller value, the superfluous feature channels are discarded. To change the
  type of features, include_pseudotensors must be specified as True (for both
  tensor and pseudotensor features) or False (for only tensor features). Similar
  to changing the maximum degree, the shape change is achieved either by padding
  with zeros or discarding the superfluous feature channels.

  Args:
    x: Input features with shape (..., S, (L+1)**2, F).
    max_degree: New max_degree (if ``None``, max_degree is auto-determined from
      the shape of ``x``)
    include_pseudotensors: If ``True``, both tensor and pseudotensor features
      are returned, if ``False``, only tensor features are returned (if
      ``None``, the type of features is auto-determined from the shape of
      ``x``).

  Returns:
    The reshaped features.
  """
  # Determine max_degree of features and whether they contain pseudotensors.
  in_max_degree = _extract_max_degree_and_check_shape(x.shape)
  input_has_pseudotensors = x.shape[-3] != 1

  # Determine desired max_degree and whether pseudotensors should be included.
  max_degree = in_max_degree if max_degree is None else max_degree
  include_pseudotensors = (
      input_has_pseudotensors
      if include_pseudotensors is None
      else include_pseudotensors
  )

  # Increase max_degree (slice).
  if max_degree < in_max_degree:
    x = x[..., : (max_degree + 1) ** 2, :]

  # Decrease max_degree (pad with zeros).
  elif max_degree > in_max_degree:
    pad_with = [(0, 0)] * x.ndim
    pad_with[-2] = (0, (max_degree + 1) ** 2 - (in_max_degree + 1) ** 2)
    x = torch.nn.functional.pad(x, pad_with, mode='constant', value=0)

  # Remove existing pseudotensor channel.
  if input_has_pseudotensors and not include_pseudotensors:
    x = torch.concat(
        [
            x[..., l % 2 : l % 2 + 1, l**2 : (l + 1) ** 2, :]
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )

  # Add non-existing pseudotensor channel.
  elif include_pseudotensors and not input_has_pseudotensors:
    # Even parity features.
    e = torch.concat(
        [
            x[..., 0:1, l**2 : (l + 1) ** 2, :]
            if l % 2 == 0
            else jnp.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )
    # Odd parity features.
    o = torch.concat(
        [
            x[..., 0:1, l**2 : (l + 1) ** 2, :]
            if l % 2 != 0
            else jnp.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )
    # Combined even and odd parity features.
    x = torch.concat((e, o), axis=-3)

  return x
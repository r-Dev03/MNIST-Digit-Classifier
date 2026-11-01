from pathlib import Path
from safetensors.numpy import save_file
import typing

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jaxtyping import Array, Scalar
from optax import OptState
from model import ModelWeights, model_forward, save_weights

RANDOM_SEED = 1234
HIDDEN_NEURONS = 32
TRAINING_STEPS = 100000
BATCH_SIZE = 256
TRAIN_SIZE = 60_000
SAVE_PATH = Path("weights.pkl")

data = np.load("mnist.npz", allow_pickle=True)

images = data["images"]
labels = data["labels"]

N_PIXELS = images.shape[-1]  # 28 * 28 = 784
N_POSSIBLE_NUMBERS = len(jnp.unique(labels))  # 10 (digits 0-9)

print("images:", images.shape)
print("labels:", labels.shape)

# Normalize the pixel values to be between 0 and 1
images = images.astype(float) / images.max()

labels = jax.nn.one_hot(labels, N_POSSIBLE_NUMBERS)
print("one-hot labels:", labels.shape)

train_images, test_images = images[:TRAIN_SIZE], images[TRAIN_SIZE:]
train_labels, test_labels = labels[:TRAIN_SIZE], labels[TRAIN_SIZE:]
print("train:", train_images.shape, "test:", test_images.shape)


rng = np.random.default_rng(RANDOM_SEED)

weights = ModelWeights(
    w1=jnp.array(
        rng.uniform(size=(N_PIXELS, HIDDEN_NEURONS)),
    ),
    b1=jnp.array(
        rng.uniform(size=(HIDDEN_NEURONS,)),
    ),
    w2=jnp.array(
        rng.uniform(size=(HIDDEN_NEURONS, N_POSSIBLE_NUMBERS)),
    ),
    b2=jnp.array(
        rng.uniform(size=(N_POSSIBLE_NUMBERS,)),
    ),
)

batch_images = train_images[:BATCH_SIZE]
batch_labels = train_labels[:BATCH_SIZE]

# print(model(weights, batch_images))


def loss_fn(weights: ModelWeights, images: Array, labels: Array) -> Scalar:
    predictions = model_forward(weights, images)
    return jnp.square(labels - predictions).mean()


optimizer = optax.adamw(
    optax.linear_schedule(
        0.005,
        0.00001,
        TRAINING_STEPS,
    )
)


@jax.jit
def training_step(
    weights: ModelWeights,
    optimizer_state: OptState,
    images: Array,
    labels: Array,
) -> tuple[ModelWeights, OptState, Scalar]:
    loss, gradients = jax.value_and_grad(loss_fn)(weights, images, labels)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, weights)
    weights = typing.cast(ModelWeights, optax.apply_updates(weights, updates))
    return weights, optimizer_state, loss


def accuracy(predictions: Array, labels: Array) -> Scalar:
    return (predictions.argmax(-1) == labels.argmax(-1)).mean()


i = 0
optimizer_state = optimizer.init(weights)
for step in range(TRAINING_STEPS):
    batch_images = train_images[i : i + BATCH_SIZE]
    batch_labels = train_labels[i : i + BATCH_SIZE]
    weights, optimizer_state, loss = training_step(
        weights,
        optimizer_state,
        batch_images,
        batch_labels,
    )
    # print(loss(weights, batch_images, batch_labels))
    if step % 1000 == 0:
        # print("Loss:", loss)
        print(
            f"Accuracy: {accuracy(model_forward(weights, batch_images), batch_labels):%}"
        )

    i += BATCH_SIZE
    if i >= len(train_images):
        i = 0

test_accuracy = accuracy(model_forward(weights, test_images), test_labels)
print(f"Test accuracy: {test_accuracy:%}")

save_weights(weights, SAVE_PATH)


save_file(
    {
        "w1": np.array(weights.w1),
        "b1": np.array(weights.b1),
        "w2": np.array(weights.w2),
        "b2": np.array(weights.b2),
    },
    "weights.safetensors",
)

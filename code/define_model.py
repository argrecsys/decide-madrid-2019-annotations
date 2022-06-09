# Get tokenizer
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from defs import *

def define_model(connect_heads=False, hidden_layers=1):
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

    # Get transformer model
    transformer_model = TFBertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", output_hidden_states=True, name="bert_for_token_classification")

    # Same size as the maximum number of tokens obtained from Autotokenizer
    input_size = tokenizer.model_max_length
    input_ids = tf.keras.Input(shape=(input_size, ), dtype="int32", name="input")

    # Get the hidden states from the model
    transformer = transformer_model(input_ids)    
    hidden_states = transformer.hidden_states
    # There are 13 hidden states

    # We take various hidden states because http://jalammar.github.io/illustrated-bert/ states that it should get a better accuracy
    # Decided to take TRANSFORMER_HIDDEN_STATES_SIZE
    hidden_states_ind = list(range(-TRANSFORMER_HIDDEN_STATES_SIZE, 0)) # [-4, -3, -2, -1]

    # Concatenate the hidden states to get one "big" KerasTensor (None x 512 x (TRANSFORMER_HIDDEN_STATES_SIZE x 768))
    selected_hidden_states = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in hidden_states_ind]), name="hidden_states_concatenation")

    # Common layers for all the outputs (None x 512 x COMMON_LAYER_SIZE)
    X = selected_hidden_states
    for i in range(hidden_layers):
        X = tf.keras.layers.Dense(COMMON_LAYER_SIZE, activation="relu", name=f"common{i}")(X)
    common = X

    # Output for BIO (Argument detection)
    ## Dense layer (None x 512 x BIO_LAYER_SIZE)
    am_bio = tf.keras.layers.Dense(BIO_LAYER_SIZE, activation="relu", name="am_bio")(common)
    ## Dense layer (None x 512 x len(BIO_NAMES))
    am_bio_output = tf.keras.layers.Dense(len(BIO_NAMES), activation="softmax", name="am_bio_output")(am_bio)

    # Output for type of argument (Premise, Claim or Major Claim) (Argument classification)
    if connect_heads:
        common = tf.keras.layers.concatenate((am_bio, common), name="bio_concatenation")
    ## Dense layer (None x 512 x AM_TYPE_LAYER_SIZE)
    am_type = tf.keras.layers.Dense(AM_TYPE_LAYER_SIZE, activation="relu", name="am_type")(common)
    ## Dense layer (None x 512 x len(ARGUMENT_TYPES_NAMES))
    am_type_output = tf.keras.layers.Dense(len(ARGUMENT_TYPES_NAMES), activation="softmax", name="am_type_output")(am_type)

    # Ouput for relation type
    if connect_heads:
        common = tf.keras.layers.concatenate((am_type, common), name="type_concatenation")
    ## Dense layer (None x 512 x REL_TYPE_LAYER_SIZE)
    am_rel_type = tf.keras.layers.Dense(REL_TYPE_LAYER_SIZE, activation="relu", name="am_rel_type")(common)
    ## Dense layer (None x 512 x len(RELATION_TYPES_NAMES))
    am_rel_type_output = tf.keras.layers.Dense(len(RELATION_TYPES_NAMES), activation="softmax", name="am_rel_type_output")(am_rel_type)

    # Ouput for relation intent
    if connect_heads:
        common = tf.keras.layers.concatenate((am_rel_type, common), name="reltype_concatenation")
    ## Dense layer (None x 512 x REL_INTENT_LAYER_SIZE)
    am_rel_intent = tf.keras.layers.Dense(REL_INTENT_LAYER_SIZE, activation="relu", name="am_rel_intent")(common)
    ## Dense layer (None x 512 x len(RELATION_INTENTS_NAMES))
    am_rel_intent_output = tf.keras.layers.Dense(len(RELATION_INTENTS_NAMES), activation="softmax", name="am_rel_intent_output")(am_rel_intent)

    # Output for relation distance
    if connect_heads:
        common = tf.keras.layers.concatenate((am_rel_intent, common), name="relintent_concatenation")
    ## Dense layer (None x 512 x REL_DISTANCE_LAYER_SIZE)
    am_rel_distance = tf.keras.layers.Dense(REL_DISTANCE_LAYER_SIZE, activation="linear", name="am_rel_distance")(common)
    ## Dense layer (None x 512 x 1)
    am_rel_distance_output = tf.keras.layers.Dense(1, activation="linear", name="am_rel_distance_output")(am_rel_distance)

    
    # Declare model
    model = tf.keras.models.Model(inputs = input_ids,
     outputs = [am_bio_output, am_type_output, am_rel_type_output, am_rel_intent_output, am_rel_distance_output],
     name="E2E_neural_model")

    for layer in model.layers[0:3]: # Bert and concat layers
        layer.trainable=False

    # Compile model
    model.compile(loss={"am_bio_output": "categorical_crossentropy",
                        "am_type_output": "categorical_crossentropy",
                        "am_rel_type_output": "categorical_crossentropy",
                        "am_rel_intent_output": "categorical_crossentropy",
                        "am_rel_distance_output": "mean_squared_error"},
                  optimizer=tf.keras.optimizers.Adam(), # Default learning rate
                  metrics={"am_bio_output": tf.keras.metrics.CategoricalAccuracy(name="am_bio_acc"),
                           "am_type_output": tf.keras.metrics.CategoricalAccuracy(name="am_type_acc"),
                           "am_rel_type_output": tf.keras.metrics.CategoricalAccuracy(name="am_rel_type_output_acc"),
                           "am_rel_intent_output": tf.keras.metrics.CategoricalAccuracy(name="am_rel_intent_output_acc"),
                           "am_rel_distance_output": tf.keras.metrics.MeanSquaredError(name="am_rel_distance_mse")})
    return model

def plot_model(model, fout):
    tf.keras.utils.plot_model(model, show_layer_activations=True, to_file=fout)

def set_bert_to_trainable(model):
    for layer in model.layers[0:3]: # Bert and concat layers
        layer.trainable=True

    model.compile(loss={"am_bio_output": "categorical_crossentropy",
                        "am_type_output": "categorical_crossentropy",
                        "am_rel_type_output": "categorical_crossentropy",
                        "am_rel_intent_output": "categorical_crossentropy",
                        "am_rel_distance_output": "mean_squared_error"},
                  optimizer=tf.keras.optimizers.Adam(1e-5), # Lower learning rate
                  metrics={"am_bio_output": tf.keras.metrics.CategoricalAccuracy(name="am_bio_acc"),
                           "am_type_output": tf.keras.metrics.CategoricalAccuracy(name="am_type_acc"),
                           "am_rel_type_output": tf.keras.metrics.CategoricalAccuracy(name="am_rel_type_output_acc"),
                           "am_rel_intent_output": tf.keras.metrics.CategoricalAccuracy(name="am_rel_intent_output_acc"),
                           "am_rel_distance_output": tf.keras.metrics.MeanSquaredError(name="am_rel_distance_mse")})
    return model
    
import os
from typing import Optional, Text, List

import tensorflow_model_analysis as tfma

from tfx import v1 as tfx
from absl import logging
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata

from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from tfx.types import Channel

from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2

from tfx.components import ExampleValidator, SchemaGen, StatisticsGen, Trainer, Pusher, Transform, Evaluator

from tfx.components import ImportExampleGen



from .config_file import (SERVING_MODEL_DIR, PIPELINE_ROOT, DATA_ROOT,
                                PIPELINE_NAME, TRAINER_MODULE, TRANSFORM_MODULE,
                                TRAIN_STEPS, METADATA_PATH, EVAL_STEPS)






def print_my():
  print(SERVING_MODEL_DIR, "Hii there there there")


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    # enable_cache: bool,
    # run_fn: Text,
    training_module_file: Text,
    transform_module_file: Text,
    data_path: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
):
    components = []

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=4),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    # Generate Example from TFRecord Files
    example_gen = ImportExampleGen(input_base=data_path, output_config=output)
    components.append(example_gen)


    # Exclude validation data since we want to produce schema based on
    # training data only
    # Computes statistics from imported data/examples
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
        exclude_splits=["eval"],
    )
    components.append(statistics_gen)

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        exclude_splits=["eval"],
        infer_feature_shape=True,
    )
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator( 
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
        exclude_splits=["eval"],
    )
    components.append(example_validator)

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_file,
    )
    components.append(transform)

    # Uses user-provided Python function that
    # implements a model e.g. YOLOV2 using Tensorflow.
    trainer = Trainer(
        module_file=training_module_file,
        # examples=example_gen.outputs["examples"],
        # Use outputs of Transform as training inputs if Transform is used.
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=train_args,
        eval_args=eval_args,
    )
    components.append(trainer)

    # The following component is experimental and may change in the future. This is
    # required to specify the latest blessed model will be used as the baseline.
    model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
    components.append(model_resolver)


    eval_config = tfma.EvalConfig(
      model_specs=[
          # This assumes a serving model with signature 'serving_default'. If
          # using estimator based EvalSavedModel, add signature_name: 'eval' and
          # remove the label_key.
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key="y_batch",
              preprocessing_function_names=['transform_features'],
              )
          ],
      metrics_specs=[
          tfma.MetricsSpec()
      ],
      slicing_specs=[
          # An empty slice spec means the overall slice, i.e. the whole dataset.
          tfma.SlicingSpec(),
          # Data can be sliced along a feature column. In this case, data is
          # sliced along feature column trip_start_hour.
          tfma.SlicingSpec(
              feature_keys=['trip_start_hour'])
      ])


    # Evaluate the model to check if it is pushable into server directory
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
    
    components.append(evaluator)


    # Push to serving directory based on evaluation analysis
    pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR)))
    components.append(pusher)

    # Finalize the pipeline from components
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        # enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


# Run pipeline based on parameters from config file
def run_pipeline():
    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT, 
        training_module_file=TRAINER_MODULE, 
        transform_module_file=TRANSFORM_MODULE, data_path=DATA_ROOT, 
        train_args=tfx.proto.TrainArgs(num_steps=TRAIN_STEPS), 
        eval_args=tfx.proto.EvalArgs(num_steps=EVAL_STEPS),
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
    )

    # RUn the entire pipeline
    LocalDagRunner().run(pipeline)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()

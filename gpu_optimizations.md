# Narato GPU server optimizations
* train.py::CellConfig -> Added class attributes BATCH_SIZE and GPU_COUNT.
    * CellConfig is the config object that is used to configure the Keras model behind maskRCNN. These parameters were not set and thus no parallel computing was done.
* maskrcnn/mrcnn/parallel_model.py::\_\_init__() -> added line *super(ParallelModel, self).\_\_init__()*
    * Setting the GPU_COUNT > 1 in the first step will create a ParallelModel which throws an error, this fixes that error.
* findBatchSize function to find optimal batch size
    * https://stackoverflow.com/a/55509704, see mrcnn/utils.py::findBatchSize() and train.py::CrystalConfig()
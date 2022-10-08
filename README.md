# traj-classification

# 1. Fetch Dataset
# 1.1 Download an specific dataset
# --> Folder with a preconceived structure, and we will store into ./datasets/[dataset]
# 1.2 yupify a given dataset
# --> create ´yupified´ subfolder into ./datasets/[dataset] + completion check

dataset: Dataset(){´path´, ´version´, ´count´, ´name´, ´classes´, ´labels´, ´trajs´ }
dataset: Dateset = load_dataset(dataset_name: str, redownload: bool = False)

# 2. Configure training

()
----------------------------------------------------------------------------------------------------------
from yuca import Dataset, Model, Feaurizer, features

# Load Dataset
d = Dataset.MnistStroke(redownload: bool=False)

# Compute features for the given dataset
featurizer = Feaurizer(selected=features.ALL, recompute: bool=False)

# Defining models
m = Model.RandomForest(featurizer=featurizer)

# Spliting dataset
train, test = d.split(0.8)

# Train the model
m.train(dataset=train, cross_validation=5)

# Evaluate the model on a test dataset
evaluation = m.evaluate(test)

----------------------------------------------------------------------------------------------------------

# Classify a single traj
label = m.predict(test[0])

# Load/Save Model
m.save(path)
label = m.load(path)

-------------------------------------------------------------------------------------------------------------


yuka.clear_cache()
  -- models/
  -- features/
  -- evaluation/
  -- .cache_dir/
  -- config.py

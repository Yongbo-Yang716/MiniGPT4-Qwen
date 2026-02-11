from lavis import tasks
from lavis.models import load_model

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)
print(len(datasets["train"]))
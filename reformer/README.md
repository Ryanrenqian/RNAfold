灵感来源：
1. Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors
文章提出在长序列上进行预训练可以提高模型在长序列下游任务的性能，这意味着，可以先使用BaseModel在完整的RNA数据集上进行预训练，然后将得到的模型进行finetune从而达到训练的目的
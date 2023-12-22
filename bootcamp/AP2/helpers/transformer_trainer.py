import transformers
import sklearn.metrics
import functools
import os


def get_model(params, db_config_base, model_nm):
    db_config = db_config_base
    if params is not None:
        db_config.update({"cls_dropout": params["cls_dropout"]})
    db_config.update({"num_labels": 2})
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_nm, config=db_config)
    return model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    classification_report = sklearn.metrics.classification_report(
        labels, predictions, target_names=["not raining", "raining"], output_dict=True
    )
    f1_not_raining = classification_report["not raining"]["f1-score"]
    f1_raining = classification_report["raining"]["f1-score"]
    return {"f1_not_raining": f1_not_raining, "f1_raining": f1_raining}


def get_trainer(dataset, db_config_base, model_nm, FOLDER_TO_OUTPUT, parameters, tokenizer):
    args = transformers.TrainingArguments(
        FOLDER_TO_OUTPUT,
        learning_rate=parameters["learning_rate"],
        warmup_ratio=parameters["warmup_ratio"],
        lr_scheduler_type=parameters["lr_scheduler_type"],
        disable_tqdm=False,
        fp16=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=parameters["batch_size"],
        per_device_eval_batch_size=parameters["batch_size"],
        num_train_epochs=parameters["epochs"],
        weight_decay=parameters["weight_decay"],
        report_to="none",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    get_model_partial = functools.partial(get_model, db_config_base=db_config_base, model_nm=model_nm)
    return transformers.Trainer(
        model_init=get_model_partial,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def run_training(parameters, model_nm, dataset, FOLDER_TO_OUTPUT, tokenizer):
    db_config_base = transformers.AutoConfig.from_pretrained(model_nm)
    os.makedirs(FOLDER_TO_OUTPUT, exist_ok=True)
    trainer = get_trainer(dataset, db_config_base, model_nm, FOLDER_TO_OUTPUT, parameters, tokenizer)
    trainer.train()

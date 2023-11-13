import collections
from matplotlib import pyplot
import numpy as np
import torch
from climbing_ai.moonboard_dataset import MoonboardDataset, extract_batch
from sklearn import metrics
import seaborn as sn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def predict(model, call_model, dataloader, device):
    predictions = []
    truths = []
    probabilities = []
    for batch in dataloader:
        compressed_batch = extract_batch(batch, device)
        logits_clsf = call_model(compressed_batch=compressed_batch, model=model)

        (
            input_ids,
            input_locations,
            masked_input_ids,
            masked_token_ids,
            masked_positions,
            attention_mask,
            grade_id,
            sequence_length,
        ) = compressed_batch

        probability = torch.nn.functional.softmax(logits_clsf, dim=-1)
        clsf_prediction = torch.argmax(probability)

        probabilities.append(probability)
        predictions.append(clsf_prediction)
        truths.append(grade_id)

    probabilities = [
        probability.squeeze().cpu().detach() for probability in probabilities
    ]
    predictions = [prediction.squeeze().cpu().detach() for prediction in predictions]
    truths = [truth.squeeze().cpu().detach() for truth in truths]

    return predictions, truths, probabilities


def calculate_metrics(truths, predictions, cls_probabilities):
    auc = roc_auc_score(
        truths, cls_probabilities, multi_class="ovo", average="weighted"
    )
    print("ROC AUC: %f" % auc)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(truths, predictions)
    print("Accuracy: %f" % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(truths, predictions, average="weighted")
    print("Precision: %f" % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(truths, predictions, average="weighted")
    print("Recall: %f" % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(truths, predictions, average="weighted")
    print("F1 score: %f" % f1)


def evaluate(model, call_model, dataloader, device):
    model.eval()
    grade_count = len(MoonboardDataset.grades)

    mae = 0
    mean_differences = []
    error_by_grade = np.zeros((grade_count, grade_count * 2 - 1))

    for batch in dataloader:
        compressed_batch = extract_batch(batch, device)
        logits_clsf = call_model(compressed_batch=compressed_batch, model=model)

        (
            input_ids,
            input_locations,
            masked_input_ids,
            masked_token_ids,
            masked_positions,
            attention_mask,
            grade_id,
            sequence_length,
        ) = compressed_batch

        clsf_prediction = torch.argmax(logits_clsf)

        diff = grade_id.cpu() - clsf_prediction.cpu()
        diff = diff.squeeze()
        mae += abs(diff)
        mean_differences.append(int(abs(diff)))
        error_by_grade[grade_id.cpu().squeeze(), diff + grade_count - 1] += 1

    model.train()

    errorByGradeRowSums = error_by_grade.sum(axis=1)
    errorByGradeNorm = error_by_grade / errorByGradeRowSums[:, np.newaxis]

    fig, ax = pyplot.subplots(dpi=150, figsize=(10, 7))
    im = ax.imshow(errorByGradeNorm, cmap=sn.light_palette("seagreen", as_cmap=True))
    ax.figure.colorbar(im)

    ax.set_xticks(np.arange(grade_count * 2 - 1))
    ax.set_xticklabels(np.arange(grade_count * 2 - 1) - (grade_count - 1))

    ax.set_yticks(np.arange(grade_count))
    ax.set_yticklabels(MoonboardDataset.grades)
    pyplot.show()

    print("MAE = {}".format(mae / len(dataloader)))

    maeDiffDictionnary = collections.Counter(mean_differences)
    maeDiffDictionnary = collections.OrderedDict(sorted(maeDiffDictionnary.items()))
    maeDiffKeyPercentage = [
        x / len(mean_differences) for x in maeDiffDictionnary.values()
    ]

    zeroGradeErrorCount = maeDiffKeyPercentage[0] * 100
    oneGradeErrorCout = maeDiffKeyPercentage[1] * 100
    correctnessCount = zeroGradeErrorCount + oneGradeErrorCout

    twoGradeErrorCount = maeDiffKeyPercentage[2] * 100
    threeGradeErrorCount = maeDiffKeyPercentage[3] * 100
    firstBoundGradeErrorCount = twoGradeErrorCount + threeGradeErrorCount

    fourGradeErrorCount = maeDiffKeyPercentage[4] * 100
    fiveGradeErrorCount = maeDiffKeyPercentage[5] * 100
    sixGradeErrorCount = maeDiffKeyPercentage[6] * 100
    secondBoundGradeErrorCount = (
        fourGradeErrorCount + fiveGradeErrorCount + sixGradeErrorCount
    )

    print(
        "Correctness Count = {{0}} : [{}%] + {{1}} : [{}%] = {}%".format(
            round(zeroGradeErrorCount, 2),
            round(oneGradeErrorCout, 2),
            round(correctnessCount, 2),
        )
    )

    print(
        "First Bound Grade Error Count = {{2}} : [{}%] + {{3}} : [{}%] = {}%".format(
            round(twoGradeErrorCount, 2),
            round(threeGradeErrorCount, 2),
            round(firstBoundGradeErrorCount, 2),
        )
    )

    print(
        "Second Bound Grade Error Count = {{4}} : [{}%] + {{5}} : [{}%] + {{6}} : [{}%] = {}%".format(
            round(fourGradeErrorCount, 2),
            round(fiveGradeErrorCount, 2),
            round(sixGradeErrorCount, 2),
            round(secondBoundGradeErrorCount, 2),
        )
    )

    pyplot.figure(dpi=150, figsize=(10, 7))
    pyplot.bar(maeDiffDictionnary.keys(), maeDiffKeyPercentage, 0.5, color="g")
    pyplot.plot()


def plot_confusion_matrix(Y_true, Y_predict, labels, title=None):
    """
    Plot the confusion matrix.
    """
    conf_matrix = metrics.confusion_matrix(Y_true, Y_predict)
    df_cm = pd.DataFrame(
        (conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)),
        index=[i for i in labels],
        columns=[i for i in labels],
    )
    pyplot.figure(dpi=150, figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap=pyplot.cm.Blues)
    pyplot.xlabel("predicted grade")
    pyplot.ylabel("actual grade")
    if title:
        pyplot.title(title)
    pyplot.show()
    return
